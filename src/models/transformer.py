"""
transformer.py — AutoregressiveTradeModel
==========================================
Encoder-decoder transformer for the NIFTY options RL agent.

Encoder input sequence (left → right in memory):
  [CLS_PORT | option_0 ... option_79 | stock_0 ... stock_29]
   token 0      tokens 1-80              tokens 81-110

  Total memory length = 1 + 80 + 30 = 111 tokens

Portfolio CLS token
  A dedicated learnable [CLS] token is prepended to the encoder sequence.
  It receives a *rich* portfolio summary via a projection layer rather than
  being part of the positional sequence — the projection is *added* to the
  CLS embedding before encoding, so the self-attention layers can route
  portfolio information throughout the whole memory.  The value head reads
  from this token (memory[:, 0, :]).

Vocabulary (90 tokens total):
  0–2   : Action   — 0=Hold, 1=Buy, 2=Sell/Close
  3–82  : Strike   — index into the 80-option chain (offset +3)
  83–87 : Lots     — 1 to 5 lots (offset +83, so token 83 → 1 lot)
  88    : <SOS>    — start-of-sequence
  89    : <EOS>    — end-of-sequence (reserved, not used in PPO rollout)

Input dimensions:
  stock_input_dim  = 10  (OHLCV + RSI + MFI + Volatility + MA_20 + ROC)
  option_input_dim = 14  (strike, instrument_type, ltp,
                          delta, gamma, vega, theta, rho,
                          holding_status, lots_holding, pnl,
                          dte, expiry_label, moneyness)
  port_input_dim   = 7   (see PortfolioEncoder docstring)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dimensions — single source of truth imported by train_rl_agent.py
# ---------------------------------------------------------------------------
STOCK_INPUT_DIM  = 10
OPTION_INPUT_DIM = 14   # +1 for vega vs. original 13
PORT_INPUT_DIM   = 7    # rich portfolio summary (see PortfolioEncoder)
VOCAB_SIZE       = 90
SOS_TOKEN        = 88
EOS_TOKEN        = 89
ACTION_RANGE     = (0, 3)    # tokens 0, 1, 2
STRIKE_RANGE     = (3, 83)   # tokens 3 … 82  → option index = token - 3
LOT_RANGE        = (83, 88)  # tokens 83 … 87 → lots = token - 83 + 1
CHAIN_SIZE       = 80        # number of options per timestep
SEQ_LEN          = 30        # stock history window


# ---------------------------------------------------------------------------
# Portfolio context vector specification (PORT_INPUT_DIM = 7)
# ---------------------------------------------------------------------------
# The TradingEnv._get_observation() must supply a 1-D vector of length 7:
#
#   idx 0 : balance / initial_balance          (normalised cash level)
#   idx 1 : total_portfolio_value / initial_balance  ← NEW CLS signal
#   idx 2 : net_delta   (sum of position deltas, sign-aware)
#   idx 3 : net_vega    (sum of position vegas,  sign-aware)
#   idx 4 : net_theta   (daily theta of the book)
#   idx 5 : num_open_positions / CHAIN_SIZE     (fraction of chain in use)
#   idx 6 : unrealised_pnl / initial_balance   (book PnL, normalised)
#
# Shape passed in:  (batch, 7)  — *no* seq dimension, the encoder adds it.
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class PortfolioEncoder(nn.Module):
    """
    Projects the 7-dimensional portfolio summary into d_model space.

    Uses a small 2-layer MLP with LayerNorm so the CLS token starts from a
    well-conditioned representation regardless of the scale of the raw inputs.
    The output is *added* to the learnable CLS embedding inside encode_state().
    """

    def __init__(self, port_input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(port_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, port_vec: torch.Tensor) -> torch.Tensor:
        """
        port_vec : (batch, port_input_dim)
        returns  : (batch, 1, d_model)  — ready to prepend to encoder sequence
        """
        return self.net(port_vec).unsqueeze(1)


class AutoregressiveTradeModel(nn.Module):
    """
    Encoder-decoder transformer for autoregressive action generation.

    Parameters
    ----------
    stock_input_dim  : int  — features per stock timestep (default 10)
    option_input_dim : int  — features per option (default 14, includes vega)
    port_input_dim   : int  — portfolio summary vector length (default 7)
    d_model          : int  — transformer hidden dimension
    nhead            : int  — number of attention heads (must divide d_model)
    num_encoder_layers : int
    num_decoder_layers : int
    dim_feedforward  : int  — FFN inner dimension in transformer layers
    dropout          : float
    """

    def __init__(
        self,
        stock_input_dim:    int   = STOCK_INPUT_DIM,
        option_input_dim:   int   = OPTION_INPUT_DIM,
        port_input_dim:     int   = PORT_INPUT_DIM,
        d_model:            int   = 128,
        nhead:              int   = 4,
        num_encoder_layers: int   = 3,
        num_decoder_layers: int   = 2,
        dim_feedforward:    int   = 512,
        dropout:            float = 0.1,
    ):
        super().__init__()
        self.d_model     = d_model
        self.vocab_size  = VOCAB_SIZE
        self._scale      = math.sqrt(d_model)

        # ------------------------------------------------------------------
        # 1. Input projections
        # ------------------------------------------------------------------
        self.stock_proj  = nn.Linear(stock_input_dim,  d_model)
        self.option_proj = nn.Linear(option_input_dim, d_model)  # 14-dim input

        # Portfolio CLS token: learnable base + projected portfolio summary
        self.cls_embedding   = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_embedding, std=0.02)
        self.portfolio_encoder = PortfolioEncoder(port_input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # ------------------------------------------------------------------
        # 2. Transformer encoder
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ------------------------------------------------------------------
        # 3. Autoregressive decoder
        # ------------------------------------------------------------------
        self.action_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        nn.init.trunc_normal_(self.action_embedding.weight, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ------------------------------------------------------------------
        # 4. Output heads
        # ------------------------------------------------------------------
        # Vocabulary head: predicts the next token in the action sequence
        self.vocab_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, VOCAB_SIZE),
        )

        # Value head: reads from the CLS token (memory[:, 0, :])
        # Estimates V(s) for PPO advantage computation
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Causal mask for the decoder
    # ------------------------------------------------------------------
    def _causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position i can only attend to j ≤ i."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------
    def encode_state(
        self,
        stock_context: torch.Tensor,   # (batch, seq_len,   stock_input_dim)
        option_chain:  torch.Tensor,   # (batch, chain_size, option_input_dim)
        port_context:  torch.Tensor,   # (batch, port_input_dim)
    ) -> torch.Tensor:
        """
        Encode the full market state into a memory tensor.

        Sequence layout in memory:
          [CLS_PORT(0) | options(1..80) | stock(81..110)]

        The CLS token carries portfolio value information and is the token
        the value head reads from.

        Returns
        -------
        memory : (batch, 1 + chain_size + seq_len, d_model)
        """
        batch = stock_context.size(0)

        # Project stock and option tokens
        stock_emb  = self.stock_proj(stock_context) * self._scale   # (B, 30, D)
        option_emb = self.option_proj(option_chain) * self._scale   # (B, 80, D)

        # Build the portfolio CLS token:
        #   learnable base + portfolio projection (sum, not concat, to stay in D)
        cls_base = self.cls_embedding.expand(batch, -1, -1)          # (B, 1, D)
        port_proj = self.portfolio_encoder(port_context)              # (B, 1, D)
        cls_token = cls_base + port_proj                              # (B, 1, D)

        # Concatenate: [CLS | options | stock]
        state_seq = torch.cat([cls_token, option_emb, stock_emb], dim=1)  # (B, 111, D)

        # Positional encoding (applied after concat so positions are meaningful)
        state_seq = self.pos_encoder(state_seq)

        memory = self.encoder(state_seq)   # (B, 111, D)
        return memory

    # ------------------------------------------------------------------
    # Value head  (reads from the CLS token at position 0)
    # ------------------------------------------------------------------
    def get_value(self, memory: torch.Tensor) -> torch.Tensor:
        """
        memory : (batch, seq, d_model)
        returns: (batch, 1)
        """
        cls_hidden = memory[:, 0, :]       # CLS token encodes portfolio state
        return self.value_head(cls_hidden)

    # ------------------------------------------------------------------
    # Decoder  (one call covers the full 3-token action sequence)
    # ------------------------------------------------------------------
    def decode_step(
        self,
        tgt_sequence: torch.Tensor,   # (batch, tgt_len)  — token ids
        memory:       torch.Tensor,   # (batch, mem_len, d_model)
    ) -> torch.Tensor:
        """
        Autoregressive decoding over the target token sequence.

        During rollout  : called step-by-step, tgt_len grows 1 → 2 → 3
        During PPO train: called once with tgt_len=3 (SOS + 2 action tokens)

        Returns
        -------
        logits : (batch, tgt_len, vocab_size)
        """
        tgt_emb  = self.action_embedding(tgt_sequence) * self._scale  # (B, T, D)
        tgt_emb  = self.pos_encoder(tgt_emb)
        tgt_mask = self._causal_mask(tgt_sequence.size(1), tgt_sequence.device)

        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
        )                                  # (B, T, D)
        logits = self.vocab_head(dec_out)  # (B, T, vocab_size)
        return logits

    # ------------------------------------------------------------------
    # Convenience: full forward (encode + decode) for supervised pre-training
    # ------------------------------------------------------------------
    def forward(
        self,
        stock_context: torch.Tensor,   # (B, seq_len,    stock_input_dim)
        option_chain:  torch.Tensor,   # (B, chain_size, option_input_dim)
        port_context:  torch.Tensor,   # (B, port_input_dim)
        tgt_sequence:  torch.Tensor,   # (B, tgt_len)
    ):
        """
        Returns
        -------
        logits : (B, tgt_len, vocab_size)
        value  : (B, 1)
        """
        memory = self.encode_state(stock_context, option_chain, port_context)
        logits = self.decode_step(tgt_sequence, memory)
        value  = self.get_value(memory)
        return logits, value


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B = 4   # batch size

    model = AutoregressiveTradeModel()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters : {total_params:,}")

    # Dummy inputs matching expected shapes
    stock   = torch.randn(B, SEQ_LEN,    STOCK_INPUT_DIM)   # (4, 30, 10)
    options = torch.randn(B, CHAIN_SIZE, OPTION_INPUT_DIM)  # (4, 80, 14)
    port    = torch.randn(B,             PORT_INPUT_DIM)     # (4,  7)
    tgt     = torch.tensor([[SOS_TOKEN, 1, 5, 84]] * B)     # (4,  4) dummy decode

    logits, value = model(stock, options, port, tgt)

    print(f"stock   shape : {stock.shape}")
    print(f"options shape : {options.shape}  ← 14 features (includes vega)")
    print(f"port    shape : {port.shape}     ← 7-dim portfolio summary")
    print(f"logits  shape : {logits.shape}   ← (B, tgt_len, {VOCAB_SIZE})")
    print(f"value   shape : {value.shape}")

    # Verify memory layout
    memory = model.encode_state(stock, options, port)
    expected_mem_len = 1 + CHAIN_SIZE + SEQ_LEN   # 111
    assert memory.shape == (B, expected_mem_len, model.d_model), \
        f"Unexpected memory shape: {memory.shape}"
    print(f"memory  shape : {memory.shape}   ← [CLS | 80 options | 30 stock]  ✓")
    print("All checks passed.")