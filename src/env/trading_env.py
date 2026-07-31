import pandas as pd
import numpy as np
import torch
import random
from utils.generate_options_data import generate_training_data, OPT_FEATURE_COLS

class TradingEnv:
    def __init__(self, stock_paths, seq_len=30, initial_balance=300000.0, use_random_walk=True):
        self.seq_len = seq_len
        self.initial_balance = initial_balance
        self.use_random_walk = use_random_walk
        self.max_episode_steps = 1000 
        
        if self.use_random_walk:
            self.stock_df = self._generate_random_walk_data(num_days=60)
        else:
            dfs = [pd.read_csv(p) for p in stock_paths]
            self.stock_df = pd.concat(dfs).sort_values('Date').reset_index(drop=True)
            
        self.stock_df = self._calculate_technical_indicators(self.stock_df)
        self.stock_data = self.stock_df[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MFI', 'Volatility', 'MA_20', 'ROC']].values.astype(np.float32)
        
        self.stock_mean = np.nanmean(self.stock_data, axis=0)
        self.stock_std = np.nanstd(self.stock_data, axis=0)
        self.stock_std[self.stock_std < 1e-6] = 1.0

        self.all_options_df = generate_training_data(self.stock_df, simulate_portfolio=False)
        self.options_by_date = dict(tuple(self.all_options_df.groupby('timestep_date')))
        
        # 14 features matching generate_options_data.py
        self.opt_cols = OPT_FEATURE_COLS 
        self.option_mean = self.all_options_df[self.opt_cols].mean().values.astype(np.float32)
        self.option_std = self.all_options_df[self.opt_cols].std().values.astype(np.float32)
        self.option_std[self.option_std < 1e-6] = 1.0

        self.reset()

    def _generate_random_walk_data(self, num_days=60):
        dates = pd.date_range(end=pd.Timestamp.now(), periods=num_days * 74, freq='5min')
        prices = 18000 * np.cumprod(1 + np.random.normal(0.0001, 0.001, len(dates)))
        return pd.DataFrame({'Date': dates, 'Open': prices, 'High': prices*1.001, 'Low': prices*0.999, 'Close': prices, 'Volume': 10000, 'Implied_Vol': 0.15})

    def _calculate_technical_indicators(self, df):
        df['RSI'] = 50.0; df['MFI'] = 50.0; df['Volatility'] = 0.01; df['MA_20'] = df['Close']; df['ROC'] = 0.0
        return df

    def reset(self):
        self.current_step = self.seq_len
        self.balance = self.initial_balance
        self.portfolio = {} # dict mapping strike_idx (0-79) to {'lots': int, 'entry_price': float}
        self.prev_portfolio_val = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        stock_window = np.clip((self.stock_data[self.current_step-self.seq_len:self.current_step] - self.stock_mean)/self.stock_std, -5, 5)
        current_date = self.stock_df.iloc[self.current_step]['Date']
        
        day_options = self.options_by_date.get(current_date, pd.DataFrame()).copy()
        
        if day_options.empty:
            opt_tensor = np.zeros((80, 14), dtype=np.float32)
            raw_options = pd.DataFrame(columns=self.opt_cols)
            net_delta = net_vega = net_theta = unrealized_pnl = total_portfolio_val = 0.0
        else:
            day_options = day_options.head(80).reset_index(drop=True)
            
            # Inject portfolio state into the raw dataframe before normalization
            net_delta = net_vega = net_theta = unrealized_pnl = 0.0
            options_val = 0.0
            
            for idx, pos in self.portfolio.items():
                if idx < len(day_options):
                    row = day_options.iloc[idx]
                    lots = pos['lots']
                    day_options.at[idx, 'holding_status'] = 1.0 if lots > 0 else -1.0
                    day_options.at[idx, 'lots_holding'] = float(abs(lots))
                    
                    pnl = (row['ltp'] - pos['entry_price']) * lots
                    day_options.at[idx, 'pnl'] = pnl
                    
                    unrealized_pnl += pnl
                    options_val += row['ltp'] * lots
                    net_delta += row['delta'] * lots
                    net_vega += row['vega'] * lots
                    net_theta += row['theta'] * lots

            total_portfolio_val = self.balance + options_val
            
            raw_vals = day_options[self.opt_cols].values.astype(np.float32)
            opt_tensor = np.clip((raw_vals - self.option_mean) / self.option_std, -5.0, 5.0)
            raw_options = day_options

        # 7-Dimensional Portfolio Context
        port_vec = np.array([
            self.balance / self.initial_balance,
            total_portfolio_val / self.initial_balance,
            net_delta,
            net_vega,
            net_theta,
            len(self.portfolio) / 80.0,
            unrealized_pnl / self.initial_balance
        ], dtype=np.float32)
        
        return {
            "stock_context": torch.tensor(stock_window).unsqueeze(0),
            "option_chain": torch.tensor(opt_tensor).unsqueeze(0),
            "port_context": torch.tensor(port_vec).unsqueeze(0), 
            "raw_options_df": raw_options,
            "portfolio_value": total_portfolio_val
        }
    
    def step(self, orders_list):
        obs = self._get_observation()
        df = obs["raw_options_df"]

        # Process multiple orders from the variable-length sequence
        if not df.empty:
            for action_type, strike_idx, lots in orders_list:
                if action_type in [1, 2] and 0 <= strike_idx < len(df):
                    ltp = df.iloc[strike_idx]['ltp']
                    
                    if action_type == 1: # Buy (Long)
                        cost = ltp * lots
                        if self.balance >= cost:
                            self.balance -= cost
                            current_lots = self.portfolio.get(strike_idx, {}).get('lots', 0)
                            self.portfolio[strike_idx] = {'lots': current_lots + lots, 'entry_price': ltp}
                            
                    elif action_type == 2: # Sell (Short to Open / Sell to Close)
                        premium_collected = ltp * lots
                        
                        # IMPORTANT: Require margin for naked shorts to prevent infinite risk blowups
                        margin_required = premium_collected * 0.15 # Rough 15% margin proxy
                        
                        current_lots = self.portfolio.get(strike_idx, {}).get('lots', 0)
                        
                        # If we have enough balance to cover the margin
                        if self.balance >= margin_required:
                            self.balance += premium_collected
                            new_lots = current_lots - lots # This can now go negative!
                            
                            if new_lots == 0:
                                del self.portfolio[strike_idx]
                            else:
                                self.portfolio[strike_idx] = {'lots': new_lots, 'entry_price': ltp}

        # Advance the environment clock
        self.current_step += 1
        done = self.current_step >= len(self.stock_df) - 1
        
        # Calculate new state and PnL reward
        next_obs = self._get_observation()
        current_portfolio_val = next_obs["portfolio_value"]
        
        reward = current_portfolio_val - self.prev_portfolio_val
        self.prev_portfolio_val = current_portfolio_val
        
        return next_obs, float(reward), done, {"portfolio_value": current_portfolio_val}