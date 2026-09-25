"""
Worst-of Phoenix Autocallable on two equity indices.

Term sheet (illustrative, loosely based on typical EUR retail notes):
  - Underlyings : EURO STOXX 50 and S&P 500, worst-of
  - Maturity    : 3 years, quarterly observations (12 dates)
  - Autocall    : if worst performance >= 100% on an observation date
                  (from the 2nd quarter), the note redeems at par + coupon
  - Coupon      : 1.40% per quarter paid if worst >= 70% (coupon barrier),
                  with memory: missed coupons are caught up later
  - Protection  : at maturity, if worst >= 60% -> par back,
                  otherwise investor gets notional * worst performance
                  (European barrier, only checked at maturity)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhoenixAutocall:
    notional: float = 1000.0
    maturity: float = 3.0
    n_obs: int = 12
    autocall_level: float = 1.00
    coupon_barrier: float = 0.70
    protection_barrier: float = 0.60
    coupon: float = 0.0140          # per period, not annualised
    memory: bool = True
    first_call: int = 2             # first observation where autocall is possible (1-based)
    names: tuple = field(default=("SX5E", "SPX"))

    @property
    def obs_times(self):
        return np.linspace(self.maturity / self.n_obs, self.maturity, self.n_obs)

    def cashflows(self, worst, r):
        """
        worst : (n_paths, n_obs) array of worst-of performance S_t / S_ref
        r     : flat discount rate (risk free + issuer credit spread)

        Returns per path PV of
          - principal leg (as fraction of notional)
          - coupon "units": PV of number of coupons received, so that
            coupon leg = notional * coupon * units
        plus some diagnostics (redemption date index, coupons paid count).

        Splitting the two legs makes the price linear in the coupon, which is
        used later to solve for the fair coupon without any root finding.
        """
        n_paths, n_obs = worst.shape
        t = self.obs_times
        df = np.exp(-r * t)

        alive = np.ones(n_paths, dtype=bool)
        missed = np.zeros(n_paths)
        principal = np.zeros(n_paths)
        units = np.zeros(n_paths)
        n_coupons = np.zeros(n_paths)
        end_idx = np.full(n_paths, n_obs - 1)

        for i in range(n_obs):
            w = worst[:, i]

            # coupon
            hit = alive & (w >= self.coupon_barrier)
            paid = (missed + 1.0) if self.memory else np.ones(n_paths)
            units += np.where(hit, paid * df[i], 0.0)
            n_coupons += np.where(hit, paid, 0.0)
            missed = np.where(hit, 0.0, missed + alive)

            if i < n_obs - 1:
                # early redemption
                if i + 1 >= self.first_call:
                    called = alive & (w >= self.autocall_level)
                    principal[called] = df[i]
                    end_idx[called] = i
                    alive &= ~called
            else:
                # maturity
                redemption = np.where(w >= self.protection_barrier, 1.0, w)
                principal[alive] = redemption[alive] * df[i]

        return principal, units, end_idx, n_coupons

    def pv_paths(self, worst, r):
        principal, units, _, _ = self.cashflows(worst, r)
        return self.notional * (principal + self.coupon * units)
