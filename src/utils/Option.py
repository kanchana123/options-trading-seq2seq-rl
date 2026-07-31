import numpy as np
from scipy.stats import norm
from typing import Literal, Dict
from datetime import datetime

class Option:
    def __init__(
        self,
        strike: float,
        option_type: Literal["CE", "PE"],
        expiry_date: datetime,
        current_date: datetime,
        underlying_price: float,
        volatility: float,
        risk_free_rate: float = 0.05,
    ):
        self.strike = strike
        self.option_type = option_type
        self.expiry_date = expiry_date
        self.current_date = current_date
        self.S = underlying_price
        self.sigma = volatility
        self.r = risk_free_rate
        
        # Portfolio state
        self.holding_status = 0.0  # 0: None, 1: Long, -1: Short
        self.lots_holding = 0.0
        self.pnl = 0.0
        self.entry_price = 0.0

        # Greeks and Price
        self.ltp = 0.0
        self.delta = 0.0
        self.gamma = 0.0
        self.theta = 0.0
        self.vega = 0.0
        self.rho = 0.0
        
        self._calculate_time_to_expiry()
        self.calculate_greeks()

    def _calculate_time_to_expiry(self):
        """Calculates T (years) based on expiry and current date."""
        delta = self.expiry_date - self.current_date
        # Convert total seconds to years (365 days)
        self.T = max(0.0, delta.total_seconds() / (365.0 * 24.0 * 3600.0))

    def update(self, S: float, current_date: datetime, sigma: float = None):
        self.S = S
        self.current_date = current_date
        self._calculate_time_to_expiry()
        if sigma is not None:
            self.sigma = sigma
        self.calculate_greeks()
        self.update_pnl()

    def calculate_greeks(self):
        if self.T <= 0 or self.sigma <= 0 or self.S <= 0:
             self.ltp = max(0.0, self.S - self.strike) if self.option_type == "CE" else max(0.0, self.strike - self.S)
             self.delta = 0.0
             self.gamma = 0.0
             self.theta = 0.0
             self.vega = 0.0
             self.rho = 0.0
             return

        d1 = (np.log(self.S / self.strike) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)
        cdf_neg_d1 = norm.cdf(-d1)
        cdf_neg_d2 = norm.cdf(-d2)

        self.gamma = pdf_d1 / (self.S * self.sigma * np.sqrt(self.T))
        # Vega is identical for CE and PE: ∂V/∂σ = S * N'(d1) * √T / 100
        # Divided by 100 to express as change per 1% move in vol (market convention)
        self.vega = self.S * pdf_d1 * np.sqrt(self.T) / 100

        if self.option_type == "CE":
            self.ltp = self.S * cdf_d1 - self.strike * np.exp(-self.r * self.T) * cdf_d2
            self.delta = cdf_d1
            self.theta = (- (self.S * pdf_d1 * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.strike * np.exp(-self.r * self.T) * cdf_d2) / 365
            self.rho = self.strike * self.T * np.exp(-self.r * self.T) * cdf_d2 / 100
        else:
            self.ltp = self.strike * np.exp(-self.r * self.T) * cdf_neg_d2 - self.S * cdf_neg_d1
            self.delta = -cdf_neg_d1
            self.theta = (- (self.S * pdf_d1 * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.strike * np.exp(-self.r * self.T) * cdf_neg_d2) / 365
            self.rho = -self.strike * self.T * np.exp(-self.r * self.T) * cdf_neg_d2 / 100

    def update_pnl(self):
        if self.lots_holding == 0:
            self.pnl = 0.0
        else:
            self.pnl = (self.ltp - self.entry_price) * self.lots_holding

    def to_dict(self) -> Dict:
        return {
            "strike": self.strike,
            "instrument_type": 1.0 if self.option_type == "CE" else 0.0,
            "ltp": self.ltp,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
            "holding_status": self.holding_status,
            "lots_holding": self.lots_holding,
            "pnl": self.pnl
        }