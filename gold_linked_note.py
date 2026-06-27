"""
gold_linked_note.py
====================
Commodity-Linked Note (CLN) Pricing & Scenario Analysis
Role: Sales – Commodities Desk | Client: Pension Fund

Product:  3-year, 100% principal-protected gold-linked note
          Payoff = Notional + Participation × max(0, (S_T - S_0)/S_0) × Notional

Model stack:
    - Black-76 for the embedded at-the-money European call on gold
    - GBM / lognormal Monte Carlo for scenario simulation
    - Finite-difference Greeks at the note level

Author:  Cheickh Diarra – Market Finance | IESEG MiM – Asset & Risk Management
"""

import numpy as np
from scipy.stats import norm


# 
# 1.  BASE PARAMETERS (market convention)
# 
DEFAULT_PARAMS = {
    "S0":            2_650.0,   # Gold spot ($/oz)
    "sigma":         0.15,      # Implied vol (15%)
    "r":             0.02,      # Risk-free rate (2%)
    "T":             3.0,       # Maturity (years)
    "participation": 0.80,      # 80% upside participation
    "notional":      100.0,     # Normalised notional ($)
    "paths":         1_000,     # MC paths for scenarios
    "seed":          42,
}


# 
# 2.  OPTION PRICING: BLACK-76
# 

def black76_call(F: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    European call price under Black-76 (log-normal forward model).

    Parameters
    
    F     : forward price of gold  F = S0 * exp(r*T) (no convenience yield)
    K     : strike price
    r     : risk-free rate (continuous compounding)
    T     : time to maturity (years)
    sigma : lognormal volatility

    Returns
    
    c  : undiscounted call price expressed per unit of forward
    """
    if T <= 0 or sigma <= 0:
        return max(F - K, 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def black76_put(F: float, K: float, r: float, T: float, sigma: float) -> float:
    """European put via put-call parity on Black-76."""
    call = black76_call(F, K, r, T, sigma)
    return call - np.exp(-r * T) * (F - K)


# 
# 3.  NOTE PRICING
# 

def price_commodity_note(
    S0:            float,
    participation: float,
    T:             float,
    r:             float,
    sigma:         float,
    K:             float | None = None,
    notional:      float = 100.0,
) -> dict:
    """
    Price the gold-linked note.

    Decomposition:
        Note Value = PV(Principal) + participation × C_Black76(S0, K, r, T, σ) × (N/S0)

    The embedded call is struck at-the-money (K = S0 by default).
    The scaling (N/S0) converts the option price ($/oz) into percentage of notional.

    Returns a dict with full decomposition (useful for pitch deck).
    """
    if K is None:
        K = S0

    F = S0 * np.exp(r * T)                         # Gold forward price
    pv_principal = notional * np.exp(-r * T)        # PV of capital guarantee
    call_per_oz = black76_call(F, K, r, T, sigma)   # Black-76 call ($/oz)

    # Scale call to % of notional
    call_scaled = call_per_oz * (notional / S0)
    note_value  = pv_principal + participation * call_scaled

    # Optionality "cost" absorbed by the issuer / spread
    optionality_cost   = participation * call_scaled
    breakeven_gold     = K * (1 + (note_value - notional) / (participation * notional))

    return {
        "note_value":       round(note_value, 4),
        "pv_principal":     round(pv_principal, 4),
        "call_per_oz":      round(call_per_oz, 4),
        "call_scaled":      round(call_scaled, 4),
        "optionality_cost": round(optionality_cost, 4),
        "forward_price":    round(F, 2),
        "breakeven_gold":   round(breakeven_gold, 2),
        "fair_value_pct":   round(note_value / notional * 100, 3),
    }


# ─
# 4.  PAYOFF FUNCTION
# 
def note_payoff(
    S0:            float,
    S_T:           np.ndarray | float,
    participation: float,
    notional:      float = 100.0,
) -> np.ndarray:
    """
    Payoff at maturity.
        Payoff = N + participation × max(0, (S_T - S_0) / S_0) × N
    """
    return notional + participation * np.maximum((S_T - S0) / S0, 0.0) * notional


# 
# 5.  MONTE CARLO SCENARIO SIMULATION
# 

def simulate_gold_paths(
    S0:             float,
    T:              float,
    r:              float,
    sigma:          float,
    paths:          int   = 1_000,
    drift_override: float | None = None,
    seed:           int   = 42,
) -> np.ndarray:
    """
    Simulate terminal gold prices under GBM (lognormal).

    S_T = S_0 × exp[(μ - σ²/2)T + σ√T · Z],  Z ~ N(0,1)

    drift_override allows real-world scenarios:
        Bull → +20%/yr
        Base → r (risk-neutral)
        Bear → -10%/yr
    """
    np.random.seed(seed)
    mu = drift_override if drift_override is not None else r
    Z  = np.random.standard_normal(paths)
    return S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_scenarios(
    S0:            float,
    participation: float,
    T:             float,
    r:             float,
    sigma:         float,
    notional:      float = 100.0,
    paths:         int   = 1_000,
    seed:          int   = 42,
) -> dict:
    """
    Run Bull / Base / Bear scenarios and compute summary statistics.

    Returns a dict keyed by scenario name with:
        - terminal gold prices array
        - payoffs array
        - expected return (%), downside probability, VaR (95%), max payoff
    """
    DRIFTS = {
        "Bull": +0.20,
        "Base":  r,       # risk-neutral
        "Bear": -0.10,
    }

    results = {}
    for scenario, mu in DRIFTS.items():
        S_T     = simulate_gold_paths(S0, T, r, sigma, paths, drift_override=mu, seed=seed)
        payoffs = note_payoff(S0, S_T, participation, notional)

        results[scenario] = {
            "drift":            mu,
            "S_T":              S_T,
            "payoffs":          payoffs,
            "S_T_mean":         round(np.mean(S_T), 2),
            "payoff_mean":      round(np.mean(payoffs), 4),
            "payoff_std":       round(np.std(payoffs), 4),
            "expected_return":  round((np.mean(payoffs) / notional - 1) * 100, 2),
            "downside_prob":    round(np.mean(payoffs <= notional) * 100, 1),  # % at floor
            "VaR_95":           round(np.percentile(payoffs, 5), 4),            # 5th pct
            "max_payoff":       round(np.max(payoffs), 2),
        }

    return results


# 
# 6.  NOTE-LEVEL GREEKS  (finite differences)
# 

def compute_note_greeks(
    S0:            float,
    participation: float,
    T:             float,
    r:             float,
    sigma:         float,
    notional:      float = 100.0,
    dS:            float = 1.0,
    dsigma:        float = 0.01,
    dr:            float = 0.0001,
    dT:            float = 1/365,
) -> dict:
    """
    Finite-difference Greeks of the note (expressed as sensitivity per unit move).

    Delta : ∂V/∂S         (per $1 move in gold)
    Gamma : ∂²V/∂S²
    Vega  : ∂V/∂σ         (per 1 pp vol move, i.e. dsigma=0.01)
    Rho   : ∂V/∂r         (per 1 bp rate move, i.e. dr=0.0001)
    Theta : -∂V/∂T        (per day, sign convention: positive = daily decay)
    """
    K_strike = S0  # Fix strike at initial S0 for all bumps

    def val(S, sig, rate, t):
        F_ = S * np.exp(rate * t)
        pv = notional * np.exp(-rate * t)
        c  = black76_call(F_, K_strike, rate, t, sig)
        return pv + participation * c * (notional / S)

    v0    = val(S0,    sigma,        r,    T)
    v_up  = val(S0+dS, sigma,        r,    T)
    v_dn  = val(S0-dS, sigma,        r,    T)
    v_vu  = val(S0,    sigma+dsigma, r,    T)
    v_ru  = val(S0,    sigma,        r+dr, T)
    v_td  = val(S0,    sigma,        r,    T - dT)

    return {
        "delta":  round((v_up - v_dn) / (2 * dS), 6),
        "gamma":  round((v_up - 2*v0 + v_dn) / (dS**2), 8),
        "vega":   round((v_vu - v0) / dsigma, 4),      # per 1% vol
        "rho":    round((v_ru - v0) / dr, 4),           # per 1 bp
        "theta":  round(-(v_td - v0) / dT, 4),          # per day (positive = decay)
    }


# 
# 7.  SENSITIVITY TABLES  (vol surface & participation ladder)
# 

def sensitivity_vol_rate(
    S0:            float,
    participation: float,
    T:             float,
    notional:      float = 100.0,
    vol_range:     list  = None,
    rate_range:    list  = None,
) -> dict:
    """
    2-D sensitivity table: note value vs (sigma, r).
    Returns dict with 'vols', 'rates', 'table' (2D list).
    """
    if vol_range  is None: vol_range  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    if rate_range is None: rate_range = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05]

    table = []
    for sig in vol_range:
        row = []
        for rate in rate_range:
            v = price_commodity_note(S0, participation, T, rate, sig, notional=notional)
            row.append(round(v["note_value"], 3))
        table.append(row)

    return {"vols": vol_range, "rates": rate_range, "table": table}


def sensitivity_participation(
    S0:       float,
    T:        float,
    r:        float,
    sigma:    float,
    notional: float = 100.0,
    p_range:  list  = None,
) -> dict:
    """Note value vs participation rate (50% → 100%)."""
    if p_range is None:
        p_range = [i / 100 for i in range(50, 105, 5)]

    values = [
        price_commodity_note(S0, p, T, r, sigma, notional=notional)["note_value"]
        for p in p_range
    ]
    return {
        "participation": p_range,
        "note_values":   [round(v, 4) for v in values],
    }


# 
# 8.  QUICK SELF-TEST  (run: python gold_linked_note.py)
# 

if __name__ == "__main__":
    p = DEFAULT_PARAMS

    print("=" * 60)
    print("GOLD-LINKED NOTE  –  Pricing Summary")
    print("=" * 60)

    pricing = price_commodity_note(
        p["S0"], p["participation"], p["T"], p["r"], p["sigma"], notional=p["notional"]
    )
    for k, v in pricing.items():
        print(f"  {k:<22}: {v}")

    print("\nGreeks (note level):")
    greeks = compute_note_greeks(
        p["S0"], p["participation"], p["T"], p["r"], p["sigma"], notional=p["notional"]
    )
    for k, v in greeks.items():
        print(f"  {k:<8}: {v}")

    print("\nScenario Analysis (expected returns):")
    scenarios = simulate_scenarios(
        p["S0"], p["participation"], p["T"], p["r"], p["sigma"],
        notional=p["notional"], paths=p["paths"], seed=p["seed"]
    )
    for name, res in scenarios.items():
        print(f"  {name:<5}: E[Return] = {res['expected_return']:>6.2f}%  |  "
              f"Downside prob = {res['downside_prob']:>5.1f}%  |  "
              f"VaR(95%) = {res['VaR_95']:.2f}")

    print("\nVol × Rate sensitivity table (note value):")
    sens = sensitivity_vol_rate(p["S0"], p["participation"], p["T"])
    header = "σ\\r  " + "  ".join(f"{r*100:.0f}%" for r in sens["rates"])
    print("  " + header)
    for i, sig in enumerate(sens["vols"]):
        row_str = "  ".join(f"{v:7.3f}" for v in sens["table"][i])
        print(f"  {sig*100:.0f}%  {row_str}")
