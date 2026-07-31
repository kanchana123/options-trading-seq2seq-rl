import pandas as pd
import numpy as np
from typing import List
from datetime import datetime, timedelta
import calendar
from Option import Option

class OptionChain:
    def __init__(
        self,
        underlying_price: float,
        strike_interval: float,
        current_date: datetime,
        expiries: List[str] = ["WEEKLY", "MONTHLY"],
        num_strikes_each_side: int = 10,
        volatility: float = 0.2,
        risk_free_rate: float = 0.05
    ):
        self.S = underlying_price
        self.strike_interval = strike_interval
        self.current_date = current_date
        self.expiries_list = expiries
        self.num_strikes = num_strikes_each_side
        self.sigma = volatility
        self.r = risk_free_rate
        
        self.options: List[Option] = []
        self._initialize_chain()

    def _get_expiry_dates(self) -> List[datetime]:
        """Calculates specific expiry datetimes based on WEEKLY/MONTHLY logic."""
        expiry_dates = set()
        
        # Helper: Find next Thursday (0=Mon, 3=Thu)
        def get_next_thursday(dt):
            days_ahead = 3 - dt.weekday()
            if days_ahead < 0: # Thursday has passed this week
                days_ahead += 7
            return dt + timedelta(days=days_ahead)

        for exp_type in self.expiries_list:
            if exp_type == "WEEKLY":
                # Next Thursday from current date
                expiry_dates.add(get_next_thursday(self.current_date))
            
            elif exp_type == "MONTHLY":
                # Last Thursday of the current month
                year, month = self.current_date.year, self.current_date.month
                cal = calendar.monthcalendar(year, month)
                # Get last Thursday of the month
                last_thursday_day = max(week[3] for week in cal if week[3] != 0)
                last_thursday_date = datetime(year, month, last_thursday_day)
                
                # If current date is past this month's expiry, get next month's
                if self.current_date.date() > last_thursday_date.date():
                    month += 1
                    if month > 12:
                        month = 1
                        year += 1
                    cal = calendar.monthcalendar(year, month)
                    last_thursday_day = max(week[3] for week in cal if week[3] != 0)
                    last_thursday_date = datetime(year, month, last_thursday_day)
                
                expiry_dates.add(last_thursday_date)

        # Sort dates and set time to 15:30 (Market Close)
        sorted_dates = sorted(list(expiry_dates))
        return [dt.replace(hour=15, minute=30, second=0, microsecond=0) for dt in sorted_dates]

    def _initialize_chain(self):
        """Initializes the options around ATM."""
        atm_strike = round(self.S / self.strike_interval) * self.strike_interval
        
        # Generate 20 strikes centered around ATM
        # range(-10, 10) gives 20 integers: -10 to 9
        strikes = [
            atm_strike + i * self.strike_interval
            for i in range(-self.num_strikes, self.num_strikes)
        ]
        
        target_dates = self._get_expiry_dates()
        
        self.options = []
        for expiry_date in target_dates:
            for strike in strikes:
                # Call Option
                ce = Option(strike, "CE", expiry_date, self.current_date, self.S, self.sigma, self.r)
                self.options.append(ce)
                # Put Option
                pe = Option(strike, "PE", expiry_date, self.current_date, self.S, self.sigma, self.r)
                self.options.append(pe)
            
        # Sort by Expiry, then Strike, then Type
        self.options.sort(key=lambda x: (x.expiry_date, x.strike, x.option_type))

    def update_chain(self, S: float, current_date: datetime, volatility: float = None):
        """Updates the underlying parameters and recalculates greeks."""
        self.S = S
        self.current_date = current_date
        if volatility is not None:
            self.sigma = volatility
            
        for opt in self.options:
            opt.update(self.S, self.current_date, self.sigma)

    def get_state(self) -> pd.DataFrame:
        """Returns the DataFrame of the current state of all options."""
        data = [opt.to_dict() for opt in self.options]
        return pd.DataFrame(data)

    def get_token_embeddings(self) -> np.ndarray:
        """Returns the raw feature matrix for the embedding layer."""
        df = self.get_state()
        # Ensure columns are in correct order
        cols = ["strike", "instrument_type", "ltp", "delta", "gamma", "theta", "rho", "holding_status", "lots_holding", "pnl"]
        return df[cols].values.astype(np.float32)
