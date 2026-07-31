import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..env.trading_env import TradingEnv
from ..models.transformer import AutoregressiveTradeModel

# --- GLOBAL CONFIG ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dir_path = os.path.dirname(os.path.realpath(__file__))

# Point to the specific weights saved by the delta-neutral script
BEST_MODEL_PATH = os.path.join(dir_path, "..", "trade_transformer_delta_neutral.pth")

# --- VOCABULARY CONSTANTS ---
VOCAB_SIZE = 90
SOS_TOKEN = 88
EOS_TOKEN = 89
ACTION_RANGE = (0, 3)     
STRIKE_RANGE = (3, 83)    
LOT_RANGE = (83, 88)      
MAX_ORDERS = 5
MAX_TOKENS = (MAX_ORDERS * 3) + 1  

def evaluate_delta_neutral_agent(model_path=BEST_MODEL_PATH, initial_balance=300000.0):
    print(f"Loading environment and model on {device}...")
    
    # Initialize environment
    env = TradingEnv([], use_random_walk=True, initial_balance=initial_balance)
    
    # Initialize and load model
    model = AutoregressiveTradeModel().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights file not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    
    obs = env.reset()
    done = False
    
    # --- TRACKERS ---
    portfolio_history = [initial_balance]
    delta_history = [0.0]
    action_counts = {0: 0, 1: 0, 2: 0} # Hold, Buy, Sell
    step_count = 0
    
    print("Starting deterministic evaluation episode...")
    
    while not done:
        s_ctx = obs["stock_context"].to(device)
        o_chn = obs["option_chain"].to(device)
        p_ctx = obs["port_context"].to(device)
        
        with torch.no_grad():
            memory = model.encode_state(s_ctx, o_chn, p_ctx)
            tgt_seq = torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=device)
            acts = []
            
            done_decoding = False
            d_step = 0
            
            # --- GREEDY DECODING LOOP ---
            while not done_decoding and d_step < MAX_TOKENS:
                logits = model.decode_step(tgt_seq, memory)[:, -1, :]
                mask = torch.full_like(logits, float('-inf')) 
                
                mask[:, EOS_TOKEN] = 0.0  
                
                token_idx = d_step % 3
                if token_idx == 0:
                    mask[:, ACTION_RANGE[0]:ACTION_RANGE[1]] = 0.0
                elif token_idx == 1:
                    mask[:, STRIKE_RANGE[0]:STRIKE_RANGE[1]] = 0.0
                elif token_idx == 2:
                    mask[:, LOT_RANGE[0]:LOT_RANGE[1]] = 0.0
                
                # Take the absolute most probable token (Argmax)
                masked_logits = logits + mask
                token = torch.argmax(masked_logits, dim=-1).unsqueeze(1)
                token_item = token.item()
                
                if token_item == EOS_TOKEN:
                    done_decoding = True
                else:
                    acts.append(token_item)
                    tgt_seq = torch.cat([tgt_seq, token], dim=1)
                    d_step += 1

        # --- PARSE MULTI-LEG ACTIONS ---
        orders_list = []
        for i in range(0, len(acts), 3):
            if i + 2 < len(acts):  
                a_type = acts[i]
                s_idx = acts[i+1] - STRIKE_RANGE[0]
                lts = acts[i+2] - LOT_RANGE[0] + 1
                orders_list.append((a_type, s_idx, lts))
                
                # Track actions
                if a_type in action_counts:
                    action_counts[a_type] += 1
        
        # Step environment
        next_obs, reward, done, info = env.step(orders_list)
        
        # Extract net delta for tracking
        net_delta = next_obs["port_context"][0][2].item()
        
        portfolio_history.append(info['portfolio_value'])
        delta_history.append(net_delta)
        
        obs = next_obs
        step_count += 1
        
        if step_count % 500 == 0:
            print(f"Step {step_count}... Port Val: {info['portfolio_value']:.2f} | Net Delta: {net_delta:.2f}")

    # --- METRICS CALCULATION ---
    final_value = portfolio_history[-1]
    total_pnl = final_value - initial_balance
    return_pct = (total_pnl / initial_balance) * 100
    avg_abs_delta = np.mean(np.abs(delta_history))
    
    # Calculate Max Drawdown
    peak = portfolio_history[0]
    max_dd = 0
    for value in portfolio_history:
        if value > peak: peak = value
        dd = (peak - value) / peak
        if dd > max_dd: max_dd = dd
            
    print("\n" + "="*50)
    print("DELTA-NEUTRAL AGENT BACKTEST RESULTS")
    print("="*50)
    print(f"Final Balance   : {final_value:.2f} INR")
    print(f"Total PnL       : {total_pnl:.2f} INR ({return_pct:.2f}%)")
    print(f"Max Drawdown    : {max_dd*100:.2f}%")
    print(f"Avg |Delta|     : {avg_abs_delta:.2f}")
    print(f"Action Profile  : Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
    print("="*50)
    
    # --- DUAL-AXIS PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot 1: Equity Curve
    ax1.plot(portfolio_history, color='#1f77b4', linewidth=1.5, label='Portfolio Value')
    ax1.axhline(y=initial_balance, color='red', linestyle='--', alpha=0.6, label='Initial Balance')
    ax1.set_title('Agent Equity Curve', fontsize=14)
    ax1.set_ylabel('Value (INR)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Delta Exposure
    ax2.plot(delta_history, color='#ff7f0e', linewidth=1.0, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.fill_between(range(len(delta_history)), delta_history, 0, alpha=0.2, color='#ff7f0e')
    ax2.set_title('Portfolio Net Delta Exposure', fontsize=14)
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Net Delta', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = 'delta_neutral_backtest.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nDual-axis plot saved to: {plot_path}")

if __name__ == "__main__":
    evaluate_delta_neutral_agent()