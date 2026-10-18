# Tail Risk Lab — forecasts, exposure and consequences

A separate educational portfolio project by Daniel Díaz-Bedoya. Open **`docs/risk/index.html`** directly in a browser; no server is required. The optional existing server also serves it at `/risk/`.

## Question and scope

Can a model learn useful structure while leaving consequential tail risk unresolved? Compare forecasting errors with the consequences of a simple operational decision. **The experiment does not prescribe model failure.** It includes a stable light-tailed control, a stable heavy-tailed regime, and a predeclared change. All outcomes are retained and no catastrophic observation is inserted. The presentation defaults to the day containing the largest realized observation within the selected scenario and seed, explicitly labeled as retrospective extreme-event inspection. This changes presentation only, not the generated data, model fitting or full-period evaluation.

These are synthetic positive operational loads, not observed solar irradiance. This experiment is not a result from the author's published articles and does not suggest that the Quito data is power-law distributed.

## Data-generating protocol fixed before fitting

For each seed in **42, 7, 123**, generate 751 days of hourly data (one context day + 600 training target days + 150 test target days).

`signal(t) = 1 + 0.45 sin(2πt/24) + 0.15 sin(2πt/168)`

`load(t) = signal(t) × innovation(t)`

| Scenario | Innovation law | What changes? |
|---|---|---|
| Light-tailed control | Gamma(shape=4, scale=0.25) | No change; mean 1 and variance 0.25 |
| Power-law tail | Pareto Type I, survival exponent 1.5, minimum 1/3 | No change; mean 1, infinite population variance |
| Regime change | Same Gamma history and first 75 test days, then 3 × Pareto | Mean and tail change on test day 76 |

Independent NumPy random streams generate Gamma and Pareto innovations. The same realizations support the regime comparison. The shift scenario is exactly equal to the light control before the switch and three times the Pareto realization afterward. There is no search for a bad seed, no capped/rejected draw, and no hand-placed disaster. The scenarios are chosen to investigate tail risk; they are not a random sample of all possible real-world processes.

Stable refers to unchanged seasonal structure and innovation law. The raw series has a periodic conditional mean, so it is not strictly stationary without accounting for seasonality. The infinite variance is a property of an idealized unbounded distribution. Its conditional mean exists; a finite test sample still yields a numerical RMSE, but the ordinary finite-variance interpretation does not carry over. In this independent-shock setup, even knowledge of the conditional mean would not reveal individual future innovations.

## Forecasting protocol

One daily forecast maps the prior 24 observed loads to the next 24 loads. No night removal applies to this generic operational-load example. Inputs exclude the true seasonal signal, future shocks and change indicator.

Four expanding validation splits train on the first **240, 330, 420, 510** target days and evaluate the following **90 days**. Select each model's configuration by mean fold MAE; then refit on all 600 pre-test days. Models remain fixed for the 150-day test, receiving newly observed prior-day inputs daily. Adjacent input windows may contain previous target values already observed at the issue time; future target values never enter fitting.

Models: Direct AR (standardized Ridge, alpha 1/100), Random Forest and Extra Trees (100 native multi-output trees, minimum leaf size 3/10), XGBoost (100 rounds, depth 2/4, histogram training). The dashboard documents the remaining settings. Model randomness is fixed at 42; data seeds vary. Repeat-yesterday is included as a baseline. Negative forecasts are clipped to zero. Each candidate has the same information set and outer folds, but the compact budgets do not establish optimal performance for every algorithm.

The light and shift scenarios reuse identical fitted models for efficiency and comparability. All three seeds and every test hour are saved. Seed comparison is descriptive, not a rare-event probability estimate. Future work could evaluate adaptive refitting, robust losses, calibrated quantile forecasts, or extreme-value models; none is silently claimed as evaluated here.

## Decision rule and interpretation

The visitor chooses exposure `e > 0` and hourly capacity `C > 0`. Accept an hour if `forecast × e ≤ C`; otherwise decline it. For accepted work, a breach is `actual × e > C` and overload is `max(actual × e − C, 0)`. Decisions use only forecasts. Actual values are used afterward to evaluate consequences.

The dashboard reports accepted and declined hours alongside breaches. A model that declines most activity must not appear superior merely because it has fewer breaches. Capacity resets hourly: this is **not** cumulative loss, bankruptcy, revenue, or a financial strategy. Declining is assumed feasible; opportunity costs, switching costs and income are not modeled. Reducing exposure reduces both activity and risk. A point-forecast threshold is an illustrative policy, not an endorsed safety procedure.

MAE, RMSE, largest underforecast and average underforecast among the largest 1% of realized loads are shown separately. Tail selection and the default largest-day spotlight are explicitly retrospective diagnostics. The spotlight stays tied to the largest observation even when another day is selected, and distinguishes its forecast shortfall from its exposure-adjusted capacity consequence. Model rankings may differ by metric. A low average error does not certify tail calibration, and a mean predictor is not a quantile predictor.

## Black swans and power laws

### Two worked decision outcomes

Before the interactive decision section, two cards use the same largest observation from the saved Pareto run (seed 42, day 23, hour 02): 81.6411 load units. All four fitted models forecast between 0.9333 and 1.3345. With capacity 10, exposure 0.1 produces 8.1641 units of workload and no overload; exposure 3 produces 244.9234 units of workload and 234.9234 units of unserved work. All four point forecasts permit both commitments under the stated gate. These fixed retrospective illustrations show that changing models does not rescue **this event**, not that model choice is universally irrelevant. “Loss” is an operational workload proxy, not money. Lower activity is the trade-off, and the contained outcome does not guarantee safety against larger shocks. Buttons apply the exact settings to the explorer without altering data or predictions.

Known simulated extremes are modeled tail events, not evidence of predicting a black swan. The predeclared switch is a distribution-change scenario. Actual unknown shocks can lie outside a model family altogether. The transferable skill is evaluating assumptions and exposure rather than asserting that any instability guarantees catastrophe.

For Pareto, `P(Z > z) = (xmin/z)^α` above xmin. Here α is the **survival** exponent; the density exponent is α+1. The Gamma survival formula plotted is `exp(-4z) × [1 + 4z + (4z)^2/2 + (4z)^3/6]`. The plotting range truncates only the theoretical chart, never the generated data.

Real-data power-law claims require more than a straight log–log line: fit an appropriate tail threshold, assess goodness of fit, and compare alternatives. There is no empirical power-law inference in this synthetic exercise.

References: [Clauset, Shalizi & Newman (2009)](https://epubs.siam.org/doi/abs/10.1137/070710111); [Taleb, The Black Swan](https://penguinrandomhousehighereducation.com/book/?isbn=9781984841582); [Shalizi on applied-learning failure and distribution shift](https://www.stat.cmu.edu/~cshalizi/dm/22/lectures/26/lecture-26.html).

## Reproduce

Using the repository environment:

```powershell
.\.venv\Scripts\python.exe src/risk_benchmark.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_risk.py -v
node tests/risk_engine.test.cjs
```

The script saves full-precision observations, predictions, scores, parameters and fold scores in `results/risk_report.json` and `docs/risk/data.js`. Browser controls select saved model runs and recompute the decision outcomes; they do not retrain. Runtime excludes imports/installation. No server is left running after development.
