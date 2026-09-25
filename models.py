import numpy as np


class CorrelatedGBM:
    """
    Multi-asset Black-Scholes under the risk neutral measure:
        dS_k / S_k = (r - q_k) dt + sigma_k dW_k,   d<W_j, W_k> = rho_jk dt

    Since the product only looks at the underlyings on the observation
    dates, we can sample the exact lognormal transition between those dates
    (no discretisation error, no need for a fine time grid).
    """

    def __init__(self, spot, vol, div, corr, r):
        self.spot = np.asarray(spot, dtype=float)
        self.vol = np.asarray(vol, dtype=float)
        self.div = np.asarray(div, dtype=float)
        self.corr = np.asarray(corr, dtype=float)
        self.r = r
        self.chol = np.linalg.cholesky(self.corr)

    def normals(self, n_paths, n_steps, rng, antithetic=True):
        n_assets = len(self.spot)
        if antithetic:
            half = rng.standard_normal((n_paths // 2, n_steps, n_assets))
            return np.concatenate([half, -half], axis=0)
        return rng.standard_normal((n_paths, n_steps, n_assets))

    def simulate(self, times, z):
        """
        times : observation times (without 0)
        z     : independent standard normals (n_paths, n_steps, n_assets)
        returns S of shape (n_paths, n_steps, n_assets)
        """
        dt = np.diff(np.concatenate([[0.0], times]))[None, :, None]
        zc = z @ self.chol.T
        drift = (self.r - self.div - 0.5 * self.vol ** 2) * dt
        log_inc = drift + self.vol * np.sqrt(dt) * zc
        return self.spot * np.exp(np.cumsum(log_inc, axis=1))


class CorrelatedHeston:
    """
    Two-asset Heston, each asset with its own variance process:
        dS_k / S_k = (r - q_k) dt + sqrt(v_k) dW_k
        dv_k       = kappa_k (theta_k - v_k) dt + xi_k sqrt(v_k) dB_k
        d<W_k, B_k> = rho_sv_k dt,   d<W_1, W_2> = rho_12 dt

    Vol shocks are built as B_k = rho_sv W_k + sqrt(1 - rho_sv^2) Z_k with Z_k
    independent, which keeps the full 4x4 correlation matrix valid.
    Scheme: Euler on log S with full truncation on v (Lord et al. 2010).
    """

    def __init__(self, spot, v0, kappa, theta, xi, rho_sv, div, rho12, r):
        self.spot = np.asarray(spot, dtype=float)
        self.v0 = np.asarray(v0, dtype=float)
        self.kappa = np.asarray(kappa, dtype=float)
        self.theta = np.asarray(theta, dtype=float)
        self.xi = np.asarray(xi, dtype=float)
        self.rho_sv = np.asarray(rho_sv, dtype=float)
        self.div = np.asarray(div, dtype=float)
        self.r = r
        self.chol = np.linalg.cholesky(np.array([[1.0, rho12], [rho12, 1.0]]))

    def simulate(self, times, steps_per_period, n_paths, rng):
        n_assets = len(self.spot)
        dt = times[0] / steps_per_period      # assumes evenly spaced dates
        sq_dt = np.sqrt(dt)
        rho_perp = np.sqrt(1.0 - self.rho_sv ** 2)

        log_s = np.tile(np.log(self.spot), (n_paths, 1))
        v = np.tile(self.v0, (n_paths, 1))
        out = np.empty((n_paths, len(times), n_assets))

        for i in range(len(times)):
            for _ in range(steps_per_period):
                zs = rng.standard_normal((n_paths, n_assets)) @ self.chol.T
                zv = self.rho_sv * zs + rho_perp * rng.standard_normal((n_paths, n_assets))
                vp = np.maximum(v, 0.0)
                log_s += (self.r - self.div - 0.5 * vp) * dt + np.sqrt(vp) * sq_dt * zs
                v += self.kappa * (self.theta - vp) * dt + self.xi * np.sqrt(vp) * sq_dt * zv
            out[:, i, :] = np.exp(log_s)
        return out
