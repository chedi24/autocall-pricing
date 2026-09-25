# Worst-of Phoenix Autocallable – Monte Carlo pricing

Pricing and risk analysis of a 3y worst-of Phoenix autocallable on EURO STOXX 50 / S&P 500.

## Product

| | |
|---|---|
| Underlyings | SX5E, SPX (worst-of) |
| Maturity | 3 years, quarterly observations |
| Autocall | worst >= 100% from the 2nd quarter, redeems at par |
| Coupon | 1.40% per quarter if worst >= 70%, with memory |
| Protection | 60% European barrier at maturity, below it the investor gets notional x worst performance |

## Method

- Correlated GBM, sampled exactly on the observation dates (the barrier is European so no daily grid is needed)
- Payoff split into a principal leg and a coupon leg, price = N (A + c B), which gives the fair coupon directly
- Discounting at risk free + issuer credit spread
- Variance reduction: antithetic variates and control variates (forwards and European puts with closed form BS prices)
- Engine checked against the Black-Scholes price of a vanilla put

Extra analysis: autocall probabilities per date, expected life, loss probability, payout distribution, risk neutral vs real world expected payoff, delta / gamma / vega / correlation / rate sensitivities with common random numbers, vol x correlation grid, and a Heston comparison to show the impact of skew.

## Run

```
pip install -r requirements.txt
python main.py
```

Runs in about 10 seconds. Market parameters are illustrative and set at the top of `main.py`.
