import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn.functional as F

from env.trading_env import TradingEnv
from models.transformer import AutoregressiveTradeModel

# --- GLOBAL CONFIG ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dir_path = os.path.dirname(os.path.realpath(__file__))

BEST_MODEL_PATH = os.path.join(dir_path, "..", "trade_transformer_delta_neutral.pth")

# Constants
VOCAB_SIZE = 90
SOS_TOKEN = 88
EOS_TOKEN = 89
ACTION_RANGE = (0, 3)     
STRIKE_RANGE = (3, 83)    
LOT_RANGE = (83, 88)      
MAX_TOKENS = 16  
BARS_PER_DAY = 74
DAYS_IN_WEEK = 5
WEEK_STEPS = BARS_PER_DAY * DAYS_IN_WEEK

def get_2d_attention_proxy(memory_tensor):
    """
    Calculates a 30x80 similarity matrix between Stock Context and Options Chain.
    """
    # memory_tensor shape: (1, 111, d_model)
    # Options are tokens 1 to 80. Stock are tokens 81 to 110.
    opt_mem = memory_tensor[0, 1:81, :]   # (80, d_model)
    stock_mem = memory_tensor[0, 81:111, :] # (30, d_model)
    
    # Normalize vectors to calculate Cosine Similarity
    stock_norm = F.normalize(stock_mem, p=2, dim=-1)
    opt_norm = F.normalize(opt_mem, p=2, dim=-1)
    
    # Matrix multiplication yields the (30, 80) correlation heatmap
    sim_matrix = torch.matmul(stock_norm, opt_norm.transpose(0, 1))
    
    # Shift values to be strictly positive [0, 1] for easier heatmap visualization
    heatmap_matrix = (sim_matrix + 1.0) / 2.0 
    return heatmap_matrix.cpu().numpy()

def generate_evaluation_data():
    print("Loading environment and agent for data collection...")
    env = TradingEnv([], use_random_walk=True, initial_balance=300000.0)
    model = AutoregressiveTradeModel().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval() 
    
    obs = env.reset()
    done = False
    records = []
    
    print("Simulating market and capturing 2D attention matrices...")
    while not done:
        s_ctx = obs["stock_context"].to(device)
        o_chn = obs["option_chain"].to(device)
        p_ctx = obs["port_context"].to(device)
        raw_df = obs["raw_options_df"]
        
        current_date = env.stock_df.iloc[env.current_step]['Date']
        stock_price = env.stock_df.iloc[env.current_step]['Close']
        
        with torch.no_grad():
            memory = model.encode_state(s_ctx, o_chn, p_ctx)
            attn_matrix = get_2d_attention_proxy(memory)
            
            # Reorder attention and labels: CE first then PE
            opt_labels = []
            if not raw_df.empty:
                sort_df = raw_df.copy()
                sort_df['orig_idx'] = range(len(sort_df))
                # CE (1.0) before PE (0.0) -> Sort instrument_type descending.
                # Also sub-sort by expiry and strike for better clarity.
                sort_df = sort_df.sort_values(by=['instrument_type', 'expiry_label', 'strike'], 
                                              ascending=[False, True, True])
                sort_indices = sort_df['orig_idx'].tolist()
                
                # Reorder heatmap columns (dimension 1 of the 30x80 matrix)
                attn_matrix = attn_matrix[:, sort_indices]
                
                # Generate labels in the new sorted order
                for idx in sort_indices:
                    row = raw_df.iloc[idx]
                    opt_type = "CE" if row['instrument_type'] == 1.0 else "PE"
                    strike = int(row['strike'])
                    opt_labels.append(f"T{idx+3}: {opt_type} {strike}")
            else:
                opt_labels = [f"T{i+3}: N/A" for i in range(80)]

            tgt_seq = torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=device)
            acts = []
            done_decoding = False
            d_step = 0
            
            while not done_decoding and d_step < MAX_TOKENS:
                logits = model.decode_step(tgt_seq, memory)[:, -1, :]
                mask = torch.full_like(logits, float('-inf')) 
                mask[:, EOS_TOKEN] = 0.0  
                
                token_idx = d_step % 3
                if token_idx == 0: mask[:, ACTION_RANGE[0]:ACTION_RANGE[1]] = 0.0
                elif token_idx == 1: mask[:, STRIKE_RANGE[0]:STRIKE_RANGE[1]] = 0.0
                elif token_idx == 2: mask[:, LOT_RANGE[0]:LOT_RANGE[1]] = 0.0
                
                masked_logits = logits + mask
                token = torch.argmax(masked_logits, dim=-1).unsqueeze(1)
                
                if token.item() == EOS_TOKEN: done_decoding = True
                else:
                    acts.append(token.item())
                    tgt_seq = torch.cat([tgt_seq, token], dim=1)
                    d_step += 1

        orders_list = []
        for i in range(0, len(acts), 3):
            if i + 2 < len(acts):  
                a_type, s_idx, lts = acts[i], acts[i+1] - STRIKE_RANGE[0], acts[i+2] - LOT_RANGE[0] + 1
                orders_list.append((a_type, s_idx, lts))
        
        next_obs, reward, done, info = env.step(orders_list)
        
        records.append({
            'step': env.current_step,
            'date': current_date,
            'day_of_week': current_date.weekday(),
            'stock_price': stock_price,
            'portfolio_value': info['portfolio_value'],
            'net_delta': info.get('net_delta', 0.0),
            'attn_matrix': attn_matrix,
            'opt_labels': opt_labels
        })
        obs = next_obs

    print("Slicing data to exactly 1 week ending on Expiry (Thursday)...")
    df = pd.DataFrame(records)
    
    thursdays = df[df['day_of_week'] == 3]
    if thursdays.empty: raise ValueError("No Expiry Thursday found.")
        
    target_thursday_idx = thursdays.index[-1] 
    start_idx = max(0, target_thursday_idx - WEEK_STEPS)
        
    sliced_df = df.iloc[start_idx:target_thursday_idx+1].reset_index(drop=True)
    return sliced_df

def create_attention_video(df, output_filename=f"{dir_path}/agent_2D_heatmap_week_v1_1.mp4"):
    print("Rendering 2D Heatmap video frames...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 2.5])
    
    ax_stock = fig.add_subplot(gs[0])
    ax_port = fig.add_subplot(gs[1])
    ax_attn = fig.add_subplot(gs[2])
    
    stock_line, = ax_stock.plot([], [], color='black', lw=1.5)
    port_line, = ax_port.plot([], [], color='#1f77b4', lw=2)
    
    ax_stock.set_xlim(0, len(df))
    ax_stock.set_ylim(df['stock_price'].min() * 0.995, df['stock_price'].max() * 1.005)
    ax_stock.set_title("Underlying Stock Price (NIFTY 50)", fontsize=12)
    ax_stock.grid(True, alpha=0.3)
    
    ax_port.set_xlim(0, len(df))
    ax_port.set_ylim(df['portfolio_value'].min() * 0.95, df['portfolio_value'].max() * 1.05)
    ax_port.set_title("Agent Portfolio Value", fontsize=12)
    ax_port.grid(True, alpha=0.3)
    
    # Initialize Heatmap
    init_matrix = df['attn_matrix'].iloc[0]
    im = ax_attn.imshow(init_matrix, cmap='magma', aspect='auto', vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax_attn, label='Cosine Similarity (Normalized)')
    
    ax_attn.set_title("Transformer Cross-Attention: Stock History vs. Option Chain", fontsize=12)
    ax_attn.set_ylabel("Stock History Window (t-30 to t)")
    ax_attn.set_xlabel("Option Tokens (Strike & Type)")
    
    # Set static Y-ticks for the 30 stock window
    ax_attn.set_yticks(np.arange(0, 30, 5))
    ax_attn.set_yticklabels([f"t-{30-i}" for i in range(0, 30, 5)])

    def update(frame):
        stock_line.set_data(range(frame), df['stock_price'].iloc[:frame])
        port_line.set_data(range(frame), df['portfolio_value'].iloc[:frame])
        
        # Update heatmap matrix
        current_matrix = df['attn_matrix'].iloc[frame]
        im.set_array(current_matrix)
        
        # Dynamically update X-axis labels as strikes shift
        labels = df['opt_labels'].iloc[frame]
        
        # Only show every 4th label to prevent overlapping text
        tick_indices = np.arange(0, 80, 4)
        ax_attn.set_xticks(tick_indices)
        ax_attn.set_xticklabels([labels[i] for i in tick_indices], rotation=90, fontsize=9)
        
        date_str = df['date'].iloc[frame].strftime('%Y-%m-%d %H:%M')
        delta_str = df['net_delta'].iloc[frame]
        fig.suptitle(f"Transformer Options Agent | {date_str} | Net Delta: {delta_str:.2f}", fontsize=16)
        
        return stock_line, port_line, im

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=50, blit=False)
    
    try:
        writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='RL_Agent'), bitrate=2000)
        ani.save(output_filename, writer=writer)
        print(f"SUCCESS! Video saved as {output_filename}")
    except Exception as e:
        print(f"FFmpeg failed. Falling back to GIF... ({e})")
        gif_filename = output_filename.replace(".mp4", ".gif")
        ani.save(gif_filename, writer='pillow', fps=15)
        print(f"SUCCESS! Video saved as {gif_filename}")

if __name__ == "__main__":
    df_week = generate_evaluation_data()
    create_attention_video(df_week)