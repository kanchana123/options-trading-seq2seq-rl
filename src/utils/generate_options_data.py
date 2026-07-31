import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import calendar

from OptionChain import OptionChain

# ---------------------------------------------------------------------------
# Column schema — must match TradingEnv.option_mean / option_std exactly.
# 11 raw option features + 3 derived = 14 total.
# Order: strike, instrument_type, ltp, delta, gamma, vega, theta, rho,
#        holding_status, lots_holding, pnl, dte,
#        expiry_label (0=near, 1=far),  moneyness (S/K - 1)
# ---------------------------------------------------------------------------
OPT_FEATURE_COLS = [
    "strike", "instrument_type", "ltp",
    "delta", "gamma", "vega", "theta", "rho",
    "holding_status", "lots_holding", "pnl",
    "dte", "expiry_label", "moneyness"
]

# Fixed chain size: 2 expiries × 20 strikes × 2 types = 80 options
CHAIN_SIZE = 80


def _get_next_thursday(dt: datetime) -> datetime:
    """Return the *next* Thursday on or after `dt`."""
    days_ahead = 3 - dt.weekday()   # 3 = Thursday
    if days_ahead < 0:
        days_ahead += 7
    return dt + timedelta(days=days_ahead)


def _get_thursday_after_next(dt: datetime) -> datetime:
    """Return the Thursday that follows the *next* Thursday — i.e. NEXT_WEEKLY."""
    first = _get_next_thursday(dt)
    return first + timedelta(days=7)


def _patch_option_chain_expiries(option_chain_instance, current_date: datetime):
    """
    OptionChain._get_expiry_dates() does not handle 'NEXT_WEEKLY'.
    We monkey-patch the instance's expiry list after construction so the
    chain object itself is untouched — keeping OptionChain.py stable.

    Instead of patching, we simply supply the two concrete expiry datetime
    objects directly when building the chain (see _build_chain below).
    """
    pass  # See _build_chain — we bypass expiries_list entirely.


def _build_chain(underlying_price: float,
                 strike_interval: float,
                 current_date: datetime,
                 volatility: float,
                 risk_free_rate: float) -> OptionChain:
    """
    Build a two-expiry option chain that always produces exactly 80 options:
      - Near expiry  : next Thursday          (WEEKLY)
      - Far  expiry  : Thursday after that    (NEXT_WEEKLY)

    We construct the chain with expiries=["WEEKLY", "MONTHLY"] (both
    recognised by OptionChain) and then *replace* the far expiry date
    afterwards. This is safer than forking OptionChain.py.

    However, the cleanest fix is simply to pass `expiries=["WEEKLY", "MONTHLY"]`
    and rely on the fact that for most dates the monthly Thursday is different
    from the weekly one, giving 2 expiries. But that breaks near month-end.

    Safest approach: sub-class or post-initialise. Here we directly call the
    OptionChain with expiries=["WEEKLY", "MONTHLY"] and, if WEEKLY == MONTHLY
    (i.e. the current week *is* the monthly expiry week), we fall back to
    NEXT_WEEKLY manually by rebuilding with patched dates.
    """
    # Build normally first
    chain = OptionChain(
        underlying_price=underlying_price,
        strike_interval=strike_interval,
        current_date=current_date,
        expiries=["WEEKLY", "MONTHLY"],
        num_strikes_each_side=10,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
    )

    unique_expiries = sorted(set(opt.expiry_date for opt in chain.options))

    # Happy path: two distinct expiry dates → exactly 80 options.
    if len(unique_expiries) == 2 and len(chain.options) == CHAIN_SIZE:
        return chain

    # Edge case: WEEKLY == MONTHLY (expiry week), so only one expiry was
    # generated → 40 options. Rebuild using NEXT_WEEKLY as the far leg.
    near_expiry = _get_next_thursday(current_date).replace(
        hour=15, minute=30, second=0, microsecond=0
    )
    far_expiry = _get_thursday_after_next(current_date).replace(
        hour=15, minute=30, second=0, microsecond=0
    )

    # Reconstruct chain manually using the two concrete dates.
    from .Option import Option as _Option

    atm_strike = round(underlying_price / strike_interval) * strike_interval
    strikes = [
        atm_strike + i * strike_interval
        for i in range(-10, 10)          # 20 strikes
    ]

    options = []
    for expiry_date in [near_expiry, far_expiry]:
        for strike in strikes:
            ce = _Option(strike, "CE", expiry_date, current_date,
                         underlying_price, volatility, risk_free_rate)
            pe = _Option(strike, "PE", expiry_date, current_date,
                         underlying_price, volatility, risk_free_rate)
            options.append(ce)
            options.append(pe)

    options.sort(key=lambda x: (x.expiry_date, x.strike, x.option_type))
    chain.options = options          # overwrite with the corrected list
    return chain


def _options_to_dataframe(chain: OptionChain,
                          current_date: datetime,
                          underlying_price: float) -> pd.DataFrame:
    """
    Convert the option chain to a DataFrame with all 13 feature columns
    plus metadata columns (expiry_date, timestep_date, underlying_price).
    Always returns exactly CHAIN_SIZE (80) rows in a deterministic order.
    """
    records = []
    unique_expiries = sorted(set(opt.expiry_date for opt in chain.options))

    for opt in chain.options:
        d = opt.to_dict()   # 10 cols from Option.to_dict()

        # --- DTE (days to expiry, integer) ---
        dte = max(0, (opt.expiry_date.date() - current_date.date()).days)
        d["dte"] = float(dte)

        # --- Expiry label: 0 = near-term, 1 = far-term ---
        d["expiry_label"] = 0.0 if opt.expiry_date == unique_expiries[0] else 1.0

        # --- Moneyness: (S/K) - 1, positive for ITM calls / OTM puts ---
        d["moneyness"] = (underlying_price / opt.strike) - 1.0

        # --- Metadata (not part of the 13-feature tensor) ---
        d["expiry_date"] = opt.expiry_date
        d["timestep_date"] = current_date
        d["underlying_price"] = underlying_price

        records.append(d)

    df = pd.DataFrame(records)

    # Validate shape
    if len(df) != CHAIN_SIZE:
        raise RuntimeError(
            f"Expected {CHAIN_SIZE} options per timestep, got {len(df)} "
            f"on date {current_date}. Check OptionChain construction."
        )

    return df


def _simulate_portfolio_state(chain: OptionChain,
                               rng: np.random.Generator) -> None:
    """
    Randomly assign open positions to 1–4 options in-place.
    Simulates realistic portfolio states for imitation / curriculum learning.
    Does NOT modify options with T ≤ 0 (expired).
    """
    tradeable = [opt for opt in chain.options if opt.T > 0]
    if not tradeable:
        return

    num_positions = rng.integers(1, min(5, len(tradeable) + 1))
    chosen = rng.choice(tradeable, size=num_positions, replace=False)

    for opt in chosen:
        opt.holding_status = float(rng.choice([-1.0, 1.0]))
        opt.lots_holding = float(rng.integers(1, 6)) * opt.holding_status
        # Entry price: within ±5–20% of current LTP (direction-aware)
        slippage = rng.uniform(0.05, 0.20)
        opt.entry_price = opt.ltp * (1.0 - opt.holding_status * slippage)
        opt.entry_price = max(0.01, opt.entry_price)
        opt.update_pnl()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_dummy_stock_data(filepath: str = "NIFTY50_data.csv") -> None:
    """Creates a minimal dummy NIFTY50 CSV for offline testing."""
    if os.path.exists(filepath):
        return
    print(f"Creating dummy stock data at {filepath} ...")
    dates = pd.date_range(start="2023-01-02", periods=60, freq="B")  # 60 business days
    prices = 18000.0 * np.cumprod(1.0 + np.random.normal(0.0003, 0.008, len(dates)))
    df = pd.DataFrame({
        "Date":   dates,
        "Open":   prices * np.random.uniform(0.998, 1.002, len(dates)),
        "High":   prices * np.random.uniform(1.002, 1.010, len(dates)),
        "Low":    prices * np.random.uniform(0.990, 0.998, len(dates)),
        "Close":  prices,
        "Volume": np.random.randint(800_000, 1_500_000, len(dates)),
        "Implied_Vol": np.clip(np.random.normal(0.15, 0.02, len(dates)), 0.08, 0.40),
    })
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df)} rows to {filepath}")


def generate_training_data(
    stock_data_input,
    output_path: str = None,
    start_idx: int = 0,
    end_idx: int = None,
    strike_interval: float = 50.0,
    volatility: float = 0.15,
    risk_free_rate: float = 0.065,
    simulate_portfolio: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a flat options DataFrame from stock price data.

    Parameters
    ----------
    stock_data_input : str or pd.DataFrame
        Path to a CSV with at least [Date, Close] columns, or a DataFrame.
        If a 'Implied_Vol' column is present it will be used per-row.
    output_path : str, optional
        If given, save the result to this CSV path.
    start_idx : int
        First row index to process (inclusive).
    end_idx : int, optional
        Last row index to process (exclusive). Defaults to len(stock_df).
    strike_interval : float
        Distance between adjacent strikes (default 50 for NIFTY).
    volatility : float
        Fallback annual implied volatility when 'Implied_Vol' is absent.
    risk_free_rate : float
        Annual risk-free rate for Black-Scholes (default 6.5%).
    simulate_portfolio : bool
        If True, randomly assign open positions to some options each timestep.
    random_seed : int
        Seed for reproducibility of portfolio simulation.

    Returns
    -------
    pd.DataFrame
        Flat DataFrame with columns:
        OPT_FEATURE_COLS + [expiry_date, timestep_date, underlying_price]
        Exactly CHAIN_SIZE rows per timestep_date.
    """
    # ------------------------------------------------------------------
    # 1. Load / validate stock data
    # ------------------------------------------------------------------
    if isinstance(stock_data_input, pd.DataFrame):
        stock_df = stock_data_input.copy()
        if "Date" in stock_df.columns:
            stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    else:
        try:
            stock_df = pd.read_csv(stock_data_input, parse_dates=["Date"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Stock data file not found: {stock_data_input}"
            )

    required_cols = {"Date", "Close"}
    missing = required_cols - set(stock_df.columns)
    if missing:
        raise ValueError(f"stock_data_input is missing columns: {missing}")

    stock_df = stock_df.sort_values("Date").reset_index(drop=True)

    if end_idx is None:
        end_idx = len(stock_df)

    end_idx = min(end_idx, len(stock_df))
    if start_idx >= end_idx:
        raise ValueError(f"start_idx ({start_idx}) >= end_idx ({end_idx})")

    has_iv_col = "Implied_Vol" in stock_df.columns

    # ------------------------------------------------------------------
    # 2. Generate one option chain per timestep
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_seed)
    all_frames: list[pd.DataFrame] = []

    print(f"Generating options data for {end_idx - start_idx} timesteps "
          f"(rows {start_idx}–{end_idx - 1}) ...")

    for idx in range(start_idx, end_idx):
        row = stock_df.iloc[idx]
        current_date: datetime = row["Date"].to_pydatetime() if hasattr(row["Date"], "to_pydatetime") else row["Date"]
        underlying_price: float = float(row["Close"])
        iv: float = float(row["Implied_Vol"]) if has_iv_col else volatility

        # Guard against degenerate prices
        if underlying_price <= 0 or np.isnan(underlying_price):
            print(f"  [WARN] Skipping idx={idx}: invalid underlying price {underlying_price}")
            continue

        # Build chain
        try:
            chain = _build_chain(
                underlying_price=underlying_price,
                strike_interval=strike_interval,
                current_date=current_date,
                volatility=iv,
                risk_free_rate=risk_free_rate,
            )
        except Exception as exc:
            print(f"  [WARN] Skipping idx={idx} ({current_date}): chain build failed — {exc}")
            continue

        # Optionally simulate open positions
        if simulate_portfolio:
            _simulate_portfolio_state(chain, rng)

        # Convert to DataFrame (raises on wrong chain size)
        try:
            day_df = _options_to_dataframe(chain, current_date, underlying_price)
        except RuntimeError as exc:
            print(f"  [WARN] Skipping idx={idx} ({current_date}): {exc}")
            continue

        all_frames.append(day_df)

        if (idx - start_idx + 1) % 100 == 0:
            print(f"  ... processed {idx - start_idx + 1} / {end_idx - start_idx} timesteps")

    if not all_frames:
        print("No data generated — returning empty DataFrame.")
        return pd.DataFrame(columns=OPT_FEATURE_COLS + ["expiry_date", "timestep_date", "underlying_price"])

    # ------------------------------------------------------------------
    # 3. Concatenate and tidy up
    # ------------------------------------------------------------------
    final_df = pd.concat(all_frames, ignore_index=True)

    # Ensure feature column order is deterministic
    meta_cols = ["expiry_date", "timestep_date", "underlying_price"]
    col_order = OPT_FEATURE_COLS + meta_cols
    final_df = final_df[col_order]

    # Cast feature columns to float32 for memory efficiency
    final_df[OPT_FEATURE_COLS] = final_df[OPT_FEATURE_COLS].astype(np.float32)

    print(f"Done. Total records: {len(final_df)}  "
          f"({len(final_df) // CHAIN_SIZE} timesteps × {CHAIN_SIZE} options)")

    # ------------------------------------------------------------------
    # 4. Optional save
    # ------------------------------------------------------------------
    if output_path:
        final_df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return final_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    STOCK_DATA_FILE = "NIFTY50_data.csv"
    OUTPUT_FILE = "options_data.csv"

    create_dummy_stock_data(STOCK_DATA_FILE)
    df = generate_training_data(
        stock_data_input=STOCK_DATA_FILE,
        output_path=OUTPUT_FILE,
        simulate_portfolio=True,
        random_seed=42,
    )

    print("\nSample output:")
    print(df.head(10).to_string(index=False))
    print(f"\nFeature columns  : {OPT_FEATURE_COLS}")
    print(f"Total columns    : {list(df.columns)}")
    print(f"Shape            : {df.shape}")
    print(f"Null values      : {df[OPT_FEATURE_COLS].isnull().sum().sum()}")