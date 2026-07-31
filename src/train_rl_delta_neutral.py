import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import logging

from env.trading_env import TradingEnv
from models.transformer import AutoregressiveTradeModel

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("RL_Trader_DeltaNeutral")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("training_metrics_delta_neutral.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# --- GLOBAL DEVICE CONFIG ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
logger.info(f"Training initialized on device: {device}")

device = torch.device("cpu") # Force CPU

# --- VOCABULARY CONSTANTS ---
VOCAB_SIZE = 90
SOS_TOKEN = 88
EOS_TOKEN = 89
ACTION_RANGE = (0, 3)     
STRIKE_RANGE = (3, 83)    
LOT_RANGE = (83, 88)      

# --- SEQUENCE LIMITS ---
MAX_ORDERS = 5
MAX_TOKENS = (MAX_ORDERS * 3) + 1  

class RolloutBuffer:
    def __init__(self):
        self.stock_contexts, self.option_chains, self.port_contexts = [], [], []
        self.actions, self.log_probs = [], []
        self.rewards, self.values, self.dones = [], [], []

    def clear(self):
        self.stock_contexts, self.option_chains, self.port_contexts = [], [], []
        self.actions, self.log_probs = [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, stock_ctx, opt_chain, port_ctx, action_seq, log_prob, reward, value, done):
        self.stock_contexts.append(stock_ctx.clone().detach().cpu())
        self.option_chains.append(opt_chain.clone().detach().cpu())
        self.port_contexts.append(port_ctx.clone().detach().cpu())
        self.actions.append(action_seq.clone().detach().cpu())
        self.log_probs.append(log_prob.clone().detach().cpu())
        self.rewards.append(reward)
        self.values.append(value.clone().detach().cpu())
        self.dones.append(done)

    def compute_gae(self, next_value, gamma=0.999, gae_lambda=0.95):
        advantages = []
        gae = 0
        values = [v.item() for v in self.values] + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * (1 - int(self.dones[step])) - values[step]
            gae = delta + gamma * gae_lambda * (1 - int(self.dones[step])) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

class PPOBatchDataset(Dataset):
    def __init__(self, buffer, advantages, returns):
        self.stock_ctx = torch.cat(buffer.stock_contexts)
        self.opt_chain = torch.cat(buffer.option_chains)
        self.port_ctx = torch.cat(buffer.port_contexts)
        self.actions = torch.stack(buffer.actions) 
        self.old_log_probs = torch.stack(buffer.log_probs)
        self.old_values = torch.cat(buffer.values)
        self.advantages = advantages
        self.returns = returns

    def __len__(self): return len(self.actions)
    def __getitem__(self, idx):
        return (self.stock_ctx[idx], self.opt_chain[idx], self.port_ctx[idx], 
                self.actions[idx], self.old_log_probs[idx], self.old_values[idx], 
                self.advantages[idx], self.returns[idx])

def generate_close_sequence(env, obs):
    seq = []
    if len(env.portfolio) > 0:
        strike_idx = random.choice(list(env.portfolio.keys()))
        lots = env.portfolio[strike_idx]['lots']
        action = 2 if lots > 0 else 1 
        lots_token = min(5, abs(lots)) - 1 + LOT_RANGE[0]
        seq.extend([action, strike_idx + STRIKE_RANGE[0], lots_token, EOS_TOKEN]) 
    if not seq: 
        seq.extend([EOS_TOKEN])
    return seq

def generate_expert_spread(env, obs):
    df = obs["raw_options_df"]
    if df.empty or len(df) < 5:
        return [EOS_TOKEN]
        
    atm_idx = (df['moneyness'].abs()).argmin()
    sell_strike = atm_idx
    buy_strike = min(atm_idx + 2, len(df) - 1)
    lots_token = LOT_RANGE[0] 
    
    seq = [
        2, sell_strike + STRIKE_RANGE[0], lots_token, 
        1, buy_strike + STRIKE_RANGE[0], lots_token,  
        EOS_TOKEN
    ]
    return seq

def train_ppo():
    # --- PATHS ---
    # Save models relative to this script's location (i.e., in the 'src' directory)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    MODEL_PATH = os.path.join(dir_path, "trade_transformer_delta_neutral.pth")
    BEST_MODEL_PATH = os.path.join(dir_path, "best_trade_transformer_delta_neutral.pth")

    env = TradingEnv([], use_random_walk=True, initial_balance=300000.0)
    model = AutoregressiveTradeModel().to(device)
    
    if os.path.exists(BEST_MODEL_PATH):
        logger.info(f"Resuming from best model: {BEST_MODEL_PATH}")
        try: model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        except Exception as e: logger.error(f"Error loading model: {e}")

    optimizer = optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    buffer = RolloutBuffer()
    best_pnl = -float('inf')

    for episode in range(5000):
        obs = env.reset()
        done = False
        buffer.clear()
        exploration_epsilon = max(0.0, 0.40 * (1 - episode / 1000.0))
        
        # --- EPISODE TRACKING METRICS ---
        ep_holds, ep_buys, ep_sells = 0, 0, 0
        ep_penalty_total = 0.0
        ep_abs_delta_sum = 0.0
        step_count = 0
        
        model.eval() 
        while not done:
            s_ctx, o_chn, p_ctx = obs["stock_context"].to(device), obs["option_chain"].to(device), obs["port_context"].to(device)
            
            with torch.no_grad():
                memory = model.encode_state(s_ctx, o_chn, p_ctx)
                value = model.get_value(memory).view(-1)
                
                tgt_seq = torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=device)
                acts, lps = [], []
                
                forced_seq = None
                if random.random() < 0.10:
                    forced_seq = generate_expert_spread(env, obs)
                elif random.random() < exploration_epsilon and env.portfolio:
                    forced_seq = generate_close_sequence(env, obs)
                
                done_decoding = False
                d_step = 0
                
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
                        
                    dist = Categorical(logits=logits + mask)
                    
                    if forced_seq is not None and d_step < len(forced_seq):
                        token_item = forced_seq[d_step]
                        token = torch.tensor([token_item], device=device)
                    else:
                        token = dist.sample()
                        token_item = token.item()
                    
                    if token_item == EOS_TOKEN:
                        acts.append(token_item)
                        lps.append(dist.log_prob(token))
                        done_decoding = True
                    else:
                        acts.append(token_item)
                        lps.append(dist.log_prob(token))
                        tgt_seq = torch.cat([tgt_seq, token.view(1, 1)], dim=1)
                        d_step += 1

            # --- TENSOR PADDING FIX ---
            padded_acts = acts + [EOS_TOKEN] * (MAX_TOKENS - len(acts))
            pad_lp_tensor = torch.tensor([0.0], device=device) # Fixed shape crash here
            padded_lps = lps + [pad_lp_tensor] * (MAX_TOKENS - len(lps))

            orders_list = []
            clean_acts = [a for a in acts if a != EOS_TOKEN]
            for i in range(0, len(clean_acts), 3):
                if i + 2 < len(clean_acts):  
                    a_type = clean_acts[i]
                    s_idx = clean_acts[i+1] - STRIKE_RANGE[0]
                    lts = clean_acts[i+2] - LOT_RANGE[0] + 1
                    orders_list.append((a_type, s_idx, lts))
                    
                    if a_type == 0: ep_holds += 1
                    elif a_type == 1: ep_buys += 1
                    elif a_type == 2: ep_sells += 1
            
            # Execute orders
            next_obs, raw_reward, done, info = env.step(orders_list)
            
            # =========================================================
            # REWARD SHAPING: THE DELTA NEUTRALITY REGULARIZER
            # =========================================================
            # Extract net_delta from the portfolio context vector (index 2)
            net_delta = next_obs["port_context"][0][2].item()
            
            # Tuning Parameters
            lambda_penalty = 100.0 
            safe_delta_threshold = 0.5 # Forgiveness zone
            
            # Calculate penalty
            excess_delta = max(0.0, abs(net_delta) - safe_delta_threshold)
            delta_penalty = lambda_penalty * excess_delta
            
            # Apply to raw PnL reward
            shaped_reward = raw_reward - delta_penalty
            
            # Update Episode Tracking
            ep_penalty_total += delta_penalty
            ep_abs_delta_sum += abs(net_delta)
            step_count += 1
            # =========================================================

            buffer.add(obs["stock_context"], obs["option_chain"], obs["port_context"], 
                       torch.tensor(padded_acts), torch.stack(padded_lps).sum(), shaped_reward, value, done)
            obs = next_obs
        
        # --- LOG EPISODE RESULTS ---
        final_pnl = info['portfolio_value'] - env.initial_balance
        avg_abs_delta = ep_abs_delta_sum / max(1, step_count)
        
        log_msg = (f"Episode {episode:04d} | Port Val: {info['portfolio_value']:>10.2f} | "
                   f"PnL: {final_pnl:>9.2f} | Avg |Delta|: {avg_abs_delta:>5.2f} | "
                   f"Penalty: {ep_penalty_total:>7.2f} | "
                   f"Orders -> H: {ep_holds:<3} B: {ep_buys:<4} S: {ep_sells:<4}")
        logger.info(log_msg)

        if final_pnl > best_pnl:
            best_pnl = final_pnl
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logger.info(f"  >>> New Best Model Saved! PnL: {best_pnl:.2f}")
        
        if (episode + 1) % 10 == 0:
            torch.save(model.state_dict(), MODEL_PATH)

        # --- PPO OPTIMIZATION LOOP ---
        model.train()
        with torch.no_grad():
            memory_next = model.encode_state(obs["stock_context"].to(device), obs["option_chain"].to(device), obs["port_context"].to(device))
            next_val = model.get_value(memory_next).squeeze().item() if not done else 0.0
            
        adv, ret = buffer.compute_gae(next_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        dataloader = DataLoader(PPOBatchDataset(buffer, adv, ret), batch_size=64, shuffle=True)
        
        for _ in range(4): # PPO Epochs
            for b_s, b_o, b_p, b_act, b_lp, b_val, b_adv, b_ret in dataloader:
                b_s, b_o, b_p, b_act, b_lp, b_adv, b_ret = [x.to(device) for x in [b_s, b_o, b_p, b_act, b_lp, b_adv, b_ret]]
                
                memory = model.encode_state(b_s, b_o, b_p)
                v_curr = model.get_value(memory).view(-1)
                
                tgt_in = torch.cat([torch.full((b_act.size(0), 1), SOS_TOKEN, dtype=torch.long, device=device), b_act[:, :-1]], dim=1)
                logits = model.decode_step(tgt_in, memory)
                
                mask = torch.full_like(logits, float('-inf'))
                mask[:, :, EOS_TOKEN] = 0.0  
                
                for step_idx in range(logits.size(1)):
                    token_idx = step_idx % 3
                    if token_idx == 0:
                        mask[:, step_idx, ACTION_RANGE[0]:ACTION_RANGE[1]] = 0.0
                    elif token_idx == 1:
                        mask[:, step_idx, STRIKE_RANGE[0]:STRIKE_RANGE[1]] = 0.0
                    elif token_idx == 2:
                        mask[:, step_idx, LOT_RANGE[0]:LOT_RANGE[1]] = 0.0
                
                dist = Categorical(logits=logits + mask)
                new_lp = dist.log_prob(b_act).sum(dim=1)
                
                # CRITICAL FIX: Clamp the exponent to prevent Infinity/NaN crashes
                log_ratio = torch.clamp(new_lp - b_lp, min=-80.0, max=80.0)
                ratio = torch.exp(log_ratio)
                
                loss_clip = -torch.min(ratio * b_adv, torch.clamp(ratio, 0.8, 1.2) * b_adv).mean()
                loss_val = 0.5 * (v_curr - b_ret).pow(2).mean()
                loss = loss_clip + loss_val
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

if __name__ == "__main__":
    train_ppo()