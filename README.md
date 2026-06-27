# Gold-Linked Note — Pricing & Scenario Analysis

A full pricing and risk analysis of a **3-year, 100% principal-protected Gold-Linked Structured Note**, built from a Sales desk perspective for a pension fund client.

## Product Overview

| Parameter | Value |
|---|---|
| Underlying | Gold (XAU/USD) |
| Maturity | 3 years |
| Capital protection | 100% |
| Participation rate | 80% |
| Payoff | `N + 80% × max(0, (S_T − S_0) / S_0) × N` |

The investor gets full principal back at maturity, plus 80% of any upside in gold.

---

## Model Stack

| Component | Method |
|---|---|
| Option pricing | **Black-76** (log-normal forward model) on an ATM European call |
| Scenario simulation | **GBM / Monte Carlo** (1 000 paths, Bull / Base / Bear) |
| Greeks | **Finite differences** (Delta, Gamma, Vega, Rho, Theta) |
| Sensitivity | 2-D heatmap (σ × r) + participation rate ladder |

---

## Key Results (base case: S₀ = $2 650/oz, σ = 15%, r = 2%)

**Pricing decomposition**
- PV of principal guarantee : **$94.18**
- Embedded call (scaled) : **$7.70**
- Note fair value : **≈ $101.87** (101.9% of notional)

**Greeks (note level)**
- Delta ≈ 0.0029 (per $1 move in gold)
- Vega ≈ 0.57 (per 1 pp vol move)
- Theta ≈ −0.0033 (daily decay)

**Scenario analysis (expected returns at maturity)**

| Scenario | Drift | E[Return] | Downside prob |
|---|---|---|---|
| Bull | +20%/yr | +24.6% | 5.6% |
| Base | +2%/yr (risk-neutral) | +6.5% | 34.0% |
| Bear | −10%/yr | +0.8% | 62.7% |

---

## Visualisations

| Chart | Description |
|---|---|
| `fig2_gold_paths.png` | Simulated GBM paths (Bull / Base / Bear) |
| `fig3_payoff_distributions.png` | Payoff distribution per scenario |
| `fig4_vega_profile.png` | Note Vega as a function of gold spot |
| `fig5_sensitivity_heatmap.png` | Note value vs σ and r (2-D heatmap) |
| `fig6_participation_sensitivity.png` | Note value vs participation rate (50–100%) |
| `fig7_spider_chart.png` | Risk/return radar across scenarios |

---

## Files

```
gold_linked_note.py            # Pricing library (Black-76, MC, Greeks, sensitivity)
gold_linked_note_notebook.ipynb  # Full interactive pitch deck with charts
fig*.png                       # Exported visualisations
```

---

## Quick Start

```bash
pip install numpy scipy matplotlib
python gold_linked_note.py
```

The script prints a full pricing summary, Greeks, scenario returns, and a vol × rate sensitivity table directly in the terminal.

---

## Author

**Cheickh Diarra** — IESEG School of Management, Master in Finance (Asset & Risk Management)  
Former Front-Office Treasury Intern @ BGFIBank Europe (FX & Money Market)
