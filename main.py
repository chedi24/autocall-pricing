import time

import numpy as np
from scipy.stats import norm

from models import CorrelatedGBM, CorrelatedHeston
from product import PhoenixAutocall

SEED = 42
N_PATHS = 200_000          # GBM, antithetic pairs included
N_PATHS_HESTON = 40_000
STEPS_PER_QUARTER = 13     # ~ weekly steps for Heston

# market data (illustrative, roughly in line with levels seen in 2026)
R = 0.025
CREDIT_SPREAD = 0.008                # issuer funding spread, the note is unsecured debt
SPOT = np.array([1.0, 1.0])          # work in performance terms, S_ref = 1
VOL = np.array([0.18, 0.17])
DIV = np.array([0.030, 0.013])
RHO = 0.60
ERP = 0.05                            # equity risk premium for the real-world run


def corr_matrix(rho):
    return np.array([[1.0, rho], [rho, 1.0]])


def bs_put(s, k, t, r, q, sigma):
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)


def control_variates(s, t, model):
    """
    Payoffs with a known Black-Scholes price, minus that price, so each column
    has mean zero: forwards on every observation date and 100% / 60% puts at
    maturity on each asset. The 60% put mimics the protection barrier.
    """
    cols = []
    for k in range(s.shape[2]):
        for i, ti in enumerate(t):
            cols.append(np.exp(-model.r * ti) * s[:, i, k] - model.spot[k] * np.exp(-model.div[k] * ti))
        for strike in (1.0, 0.6):
            payoff = np.exp(-model.r * t[-1]) * np.maximum(strike - s[:, -1, k], 0.0)
            cols.append(payoff - bs_put(model.spot[k], strike, t[-1], model.r, model.div[k], model.vol[k]))
    return np.column_stack(cols)


def mc_stats(pv, antithetic=True, controls=None):
    """
    Mean and standard error. Antithetic draws are averaged in pairs first,
    otherwise the SE would be understated. Controls are applied by regression
    (beta estimated on the same sample, the bias is O(1/n) and negligible here).
    """
    if antithetic:
        half = len(pv) // 2
        pv = 0.5 * (pv[:half] + pv[half:])
        if controls is not None:
            controls = 0.5 * (controls[:half] + controls[half:])
    if controls is not None:
        xc = controls - controls.mean(axis=0)
        beta = np.linalg.lstsq(xc, pv - pv.mean(), rcond=None)[0]
        pv = pv - controls @ beta
    return pv.mean(), pv.std(ddof=1) / np.sqrt(len(pv))


def price_gbm(note, model, z, spread=CREDIT_SPREAD):
    s = model.simulate(note.obs_times, z)
    return note.pv_paths(s.min(axis=2), model.r + spread)


def total_received(note, worst):
    """undiscounted cash received over the life of the note, per path"""
    _, _, end_idx, n_cpn = note.cashflows(worst, 0.0)
    wT = worst[:, -1]
    called = end_idx < note.n_obs - 1
    redemption = np.where(called | (wT >= note.protection_barrier), 1.0, wT)
    return note.notional * (redemption + note.coupon * n_cpn), called, wT


def section(title):
    print("\n" + title)
    print("-" * len(title))


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(SEED)
    note = PhoenixAutocall()
    model = CorrelatedGBM(SPOT, VOL, DIV, corr_matrix(RHO), R)
    disc = R + CREDIT_SPREAD
    t = note.obs_times
    z = model.normals(N_PATHS, note.n_obs, rng)

    print("Worst-of Phoenix Autocallable  |  %s / %s  |  %dy, %d obs"
          % (*note.names, note.maturity, note.n_obs))
    print("Autocall %.0f%%, coupon barrier %.0f%%, protection %.0f%%, coupon %.2f%% p.q. (memory)"
          % (100 * note.autocall_level, 100 * note.coupon_barrier,
             100 * note.protection_barrier, 100 * note.coupon))
    print("r = %.2f%%, issuer spread = %.0fbp, vols = %s, divs = %s, rho = %.2f"
          % (100 * R, 1e4 * CREDIT_SPREAD, VOL, DIV, RHO))

    # 1. sanity check of the engine against Black-Scholes
    section("1. Engine check: 3y ATM put on %s" % note.names[0])
    s = model.simulate(t, z)
    put_mc = np.exp(-R * note.maturity) * np.maximum(1.0 - s[:, -1, 0], 0.0)
    m, se = mc_stats(put_mc)
    exact = bs_put(1.0, 1.0, note.maturity, R, DIV[0], VOL[0])
    print("MC %.5f +/- %.5f   BS %.5f   diff %.1f SE" % (m, se, exact, (m - exact) / se))

    # 2. price = discounted risk neutral expectation of the payoff
    section("2. Fair value (risk neutral)")
    worst = s.min(axis=2)
    principal, units, end_idx, n_cpn = note.cashflows(worst, disc)
    pv = note.notional * (principal + note.coupon * units)
    cv = control_variates(s, t, model)
    price, se = mc_stats(pv, controls=cv)
    print("PV = %.2f  (%.2f%% of notional)   SE %.3f   95%% CI [%.2f, %.2f]"
          % (price, 100 * price / note.notional, se, price - 1.96 * se, price + 1.96 * se))
    print("Sold at par -> issuer margin ~ %.2f%% of notional" % (100 * (1 - price / note.notional)))

    # variance reduction, same number of paths each time
    z_plain = model.normals(N_PATHS, note.n_obs, np.random.default_rng(SEED + 1), antithetic=False)
    _, se_plain = mc_stats(price_gbm(note, model, z_plain), antithetic=False)
    _, se_anti = mc_stats(pv)
    print("SE with %dk paths:  plain %.3f | antithetic %.3f | antithetic + controls %.3f"
          % (N_PATHS // 1000, se_plain, se_anti, se))
    print("Variance reduction vs plain: antithetic x%.2f, with controls x%.2f"
          % ((se_plain / se_anti) ** 2, (se_plain / se) ** 2))

    print("Convergence:", end="")
    half = N_PATHS // 2
    for n in (10_000, 50_000, 100_000, N_PATHS):
        idx = np.r_[0:n // 2, half:half + n // 2]
        m, e = mc_stats(pv[idx], controls=cv[idx])
        print("  %dk: %.2f (%.2f)" % (n // 1000, m, e), end="")
    print()

    # fair coupon: price is linear in the coupon -> PV = N * (A + c * B)
    a, b = principal.mean(), units.mean()
    c_fair = (1.0 - a) / b
    c_margin = (0.98 - a) / b
    print("Fair coupon at par: %.3f%% p.q. (%.2f%% p.a.);  keeping a 2%% margin: %.3f%% p.q."
          % (100 * c_fair, 400 * c_fair, 100 * c_margin))

    # 3. scenario breakdown
    section("3. Life of the note")
    called = end_idx < note.n_obs - 1
    print("Obs  t(y)   P(redeem here)     cum")
    cum = 0.0
    for i in range(note.first_call - 1, note.n_obs):
        p = np.mean(end_idx == i) if i < note.n_obs - 1 else 1.0 - called.mean()
        cum += p
        print("%3d  %4.2f   %6.2f%%        %6.2f%%" % (i + 1, t[i], 100 * p, 100 * cum))

    total, called, wT = total_received(note, worst)
    loss = ~called & (wT < note.protection_barrier)
    print("Expected life: %.2f y,  avg number of coupons: %.2f" % (t[end_idx].mean(), n_cpn.mean()))
    print("P(autocall) %.2f%%   P(par at maturity) %.2f%%   P(capital loss) %.2f%%"
          % (100 * called.mean(), 100 * np.mean(~called & ~loss), 100 * loss.mean()))
    print("Avg redemption given loss: %.1f%% of notional" % (100 * wT[loss].mean()))

    # real world: same shocks, drift r + ERP instead of r (discounting unchanged)
    model_p = CorrelatedGBM(SPOT, VOL, DIV - ERP, corr_matrix(RHO), R)
    w_p = model_p.simulate(t, z).min(axis=2)
    total_p, called_p, wT_p = total_received(note, w_p)
    print("E[cash received], undiscounted:  risk neutral %.2f  |  real world (ERP %.0f%%) %.2f"
          % (total.mean(), 100 * ERP, total_p.mean()))
    print("P(capital loss) real world: %.2f%%"
          % (100 * np.mean(~called_p & (wT_p < note.protection_barrier))))
    print("Payout distribution (risk neutral):")
    bins = [0, 600, 900, 1000, 1050, 1100, 2000]
    hist, _ = np.histogram(total, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], hist):
        print("  [%4d, %4d)  %6.2f%%" % (lo, hi, 100 * c / len(total)))

    # 4. greeks with common random numbers
    section("4. Sensitivities (bump & revalue, same random numbers)")

    def reprice(spot=SPOT, vol=VOL, rho=RHO, r=R):
        m_ = CorrelatedGBM(spot, vol, DIV, corr_matrix(rho), r)
        return price_gbm(note, m_, z).mean()

    base = pv.mean()      # same estimator as the bumped prices, so noise cancels
    h = 0.02              # payoff has jumps at barriers, too small a bump is noisy
    for k, name in enumerate(note.names):
        up, dn = SPOT.copy(), SPOT.copy()
        up[k] += h
        dn[k] -= h
        p_up, p_dn = reprice(spot=up), reprice(spot=dn)
        delta = (p_up - p_dn) / (2 * h) / note.notional
        gamma = (p_up - 2 * base + p_dn) / h ** 2 / note.notional
        print("%-5s delta %+.3f   gamma %+.3f   (per unit of notional)" % (name, delta, gamma))
    print("Vega, parallel +1 vol pt: %+.2f" % ((reprice(vol=VOL + 0.01) - reprice(vol=VOL - 0.01)) / 2))
    for k, name in enumerate(note.names):
        bump = VOL.copy()
        bump[k] += 0.01
        print("  vega %-5s: %+.2f" % (name, reprice(vol=bump) - base))
    print("Correlation +0.05: %+.2f" % (reprice(rho=RHO + 0.05) - base))
    print("Rates +10bp:       %+.2f" % (reprice(r=R + 0.001) - base))

    print("\nPrice (% of notional) vs vol shift and correlation")
    rhos = (0.3, 0.6, 0.9)
    print("vol shift " + "".join("   rho=%.1f" % x for x in rhos))
    for dv in (-0.05, 0.0, 0.05):
        row = "".join("   %7.2f" % (100 * reprice(vol=VOL + dv, rho=x) / note.notional) for x in rhos)
        print("  %+.2f   %s" % (dv, row))

    # 5. model risk: Heston with the same long run vol but a skew
    section("5. Model risk: Heston vs Black-Scholes")
    heston = CorrelatedHeston(
        spot=SPOT, v0=VOL ** 2, kappa=[2.0, 2.0], theta=VOL ** 2, xi=[0.5, 0.5],
        rho_sv=[-0.7, -0.7], div=DIV, rho12=RHO, r=R,
    )
    s_h = heston.simulate(t, STEPS_PER_QUARTER, N_PATHS_HESTON, rng)
    ph, seh = mc_stats(note.pv_paths(s_h.min(axis=2), disc), antithetic=False)
    print("Heston PV %.2f (%.2f%%) +/- %.2f   vs BS %.2f" % (ph, 100 * ph / note.notional, seh, price))
    df_T = np.exp(-R * note.maturity)
    for strike in (1.0, 0.6):
        put_h = df_T * np.maximum(strike - s_h[:, -1, 0], 0.0).mean()
        put_bs = bs_put(1.0, strike, note.maturity, R, DIV[0], VOL[0])
        print("  3y %3.0f%% put on %s: Heston %.4f  BS %.4f" % (100 * strike, note.names[0], put_h, put_bs))
    print("Negative spot/vol correlation fattens the left tail, so the 60% put the")
    print("investor is short is worth more and the note less than under flat vol.")

    print("\nRuntime: %.1fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()
