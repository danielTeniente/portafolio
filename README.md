# Time Series & Decision Risk

**Daniel Díaz-Bedoya** · [Research profile / ORCID](https://orcid.org/0009-0000-3222-6037)

Two applied projects demonstrate how I turn time-series data into forecasts, evaluate those forecasts honestly, and examine the consequences of acting on them. The solar project builds on my published research. The separate Tail Risk Lab explores why a useful prediction can still leave a decision exposed to severe losses.

**The main findings:** a regularized linear model outperformed the tested neural networks on the Quito benchmark; in the tail-risk worked example, all four classical models underestimated the same extreme, while changing exposure changed the outcome from no overload to severe capacity failure.

## Explore the two projects

| Project | Question | Start here |
|---|---|---|
| **Quito Solar** · real observations | Does a more complex model improve tomorrow’s daylight forecast enough to justify its complexity? | [Dashboard](docs/index.html) · [Full methodology](SOLAR_FORECASTING.md) |
| **Tail Risk Lab** · synthetic experiment | What happens when a forecast becomes a commitment, and an extreme exceeds expectations? | [Dashboard](docs/risk/index.html) · [Full methodology](RISK_LAB.md) |

Download or clone the repository and open either HTML file in your browser. Both dashboards work offline without a server or build step. GitHub’s repository view displays the HTML source; open the downloaded file to interact with it.

## 1. Quito Solar — forecasting with a fair comparison

**Planning question:** can yesterday’s solar-irradiance observations improve tomorrow’s daylight forecast compared with simply repeating yesterday?

The dataset contains **12,360 hourly observations from February 2017 through June 2018**, attributed to Quito’s REMMAQ monitoring network. Each daily forecast uses the previous calendar day to predict the next day. Night hours are excluded: the retained 06:00–18:00 window gives **13 daylight inputs and 13 daylight targets**, preserving the real overnight boundary.

Direct AR, Random Forest, Extra Trees, XGBoost, RNN, GRU and LSTM are compared with a repeat-yesterday baseline. Four expanding chronological validation folds select configurations; the final evaluation covers **103 held-out days**. Preprocessing is fitted using training data only. Compact searches, multi-output models and small CPU-trained recurrent networks keep the experiment practical to reproduce.

### Main results

Recorded test MAE in source RS units; lower is better:

| Model | Test MAE | Improvement over repeat yesterday |
|---|---:|---:|
| **Direct AR** | **109.96** | **16.7%** |
| Extra Trees | 111.96 | 15.2% |
| GRU | 112.15 | 15.1% |
| Random Forest | 113.14 | 14.3% |
| XGBoost | 113.57 | 14.0% |
| LSTM | 120.35 | 8.8% |
| RNN | 122.48 | 7.2% |
| Repeat yesterday | 132.03 | — |

**Added complexity did not improve accuracy in this experiment.** Direct AR was selected by validation and also achieved the lowest test MAE. GRU, the strongest recurrent model by validation, had 2.0% higher test error; LSTM and RNN had 9.5% and 11.4% higher error. None of the three tested neural initializations beat Direct AR. The neural models still improved on persistence, but their extra training and maintenance machinery brought no observed accuracy gain over the strongest simpler model.

![Solar forecast accuracy and training cost](docs/complexity.png)

This supports choosing Direct AR **for this dataset, information window and search budget**. It does not establish universal superiority, statistical significance or monetary savings. The neural extension reused an already-inspected test period; fresh external data is needed for independent confirmation.

### What this demonstrates

- **Forecast design:** define the issue time, available information and target horizon before choosing a model; remove nights without inventing a continuous daylight-only timeline.
- **Leakage prevention:** preserve chronology and fit scaling, daylight selection and neural early stopping within the appropriate training boundaries.
- **Stability assessment:** compare four time folds and report sample counts. Each validation window contains only 51–52 days, so seasonal mix and serial dependence can move scores; fold variation is descriptive, not proof of instability or significance.
- **Model judgment:** compare against a simple baseline, measure accuracy alongside training cost, and make architectures and tuning budgets inspectable.
- **Metric interpretation:** distinguish average error, bias and difficult days; avoid MAPE near zero and avoid translating irradiance error directly into energy or financial savings.

The dashboard includes every held-out daily profile, fold results, neural seed sensitivity, training costs and an architecture information section. [Saved metrics](results/metrics.csv), [validation scores](results/validation.csv) and the [full report](results/report.json) make the conclusions auditable.

## 2. Tail Risk Lab — from prediction error to decision consequences

**Decision question:** can a model learn recurring structure while still failing to protect a commitment from an extreme event?

Direct AR, Random Forest, Extra Trees and XGBoost forecast the next 24 hourly loads from the previous 24, with persistence as a baseline. The experiment covers a light-tailed control, a stable power-law regime and a predeclared regime change. It uses **three predeclared data seeds, four expanding validation folds and 150 test days per scenario and seed**. Neural networks are excluded from this experiment.

**The data is generated before model fitting.** No disaster is inserted, no seed is chosen for poor performance, and no benign outcome is discarded. The scenarios deliberately investigate tail risk; they do not estimate how common such failures are in the real world. The largest-observation spotlight is a retrospective diagnostic, while all days and full-period scores remain available.

### Main result: the same miss, two different outcomes

In the saved power-law run with seed 42, the largest observed load is **81.64 units**. At that hour, all four fitted models forecast only **0.93–1.33 units**. The illustrative decision rule accepts work when `forecast × exposure ≤ hourly capacity`.

With capacity fixed at 10 units, every model permits both commitments:

| Decision at the same observed event | Limited exposure | High exposure |
|---|---:|---:|
| Exposure multiplier | 0.1× | 3× |
| Actual workload | 8.16 | 244.92 |
| Hourly capacity | 10.00 | 10.00 |
| **Unserved workload** | **0.00** | **234.92** |
| Outcome | The miss is absorbed | Workload greatly exceeds capacity |

**Switching between these four models does not rescue this event: they all underestimate it and accept the work. Limiting exposure changes the consequence.** This is a specific observed result, not a claim that model choice never matters. Less exposure also means less activity, and the contained outcome does not guarantee protection against a larger future shock.

“Loss” here means unserved workload for accepted commitments, not money, cumulative loss or bankruptcy. The dashboard also reports declined hours so that refusing activity cannot masquerade as better forecasting. The rule illustrates a vulnerability of relying on point forecasts; it is not a validated safety policy.

### What this demonstrates

- **Tail risk versus distribution change:** an unchanged heavy-tailed process can produce extremes; nonstationarity is a separate risk and does not automatically imply catastrophe.
- **Statistical limits:** distinguish a finite mean from infinite population variance, and distinguish mean forecasts from calibrated tail probabilities or quantiles.
- **Decision design:** evaluate exposure, capacity, accepted work and overload alongside MAE. A forecast miss becomes consequential through the commitment built around it.
- **Stress-test integrity:** retain controls, all seeds and all outcomes; label retrospective extremes rather than presenting them as representative days or predictions.
- **Black-swan reasoning:** simulated extremes belong to a known model family. They do not demonstrate prediction of unknown shocks. The practical skill is questioning what lies outside the model and preserving the ability to stop, adapt or absorb a miss.

The dashboard lets visitors change models, regimes, seeds, days, exposure and capacity. It shows average error, tail shortfall, before/after comparisons, all-seed results and exact model architectures. Controls use saved forecasts and recompute consequences without retraining. See the [complete protocol](RISK_LAB.md) and [saved observations and predictions](results/risk_report.json).

## Research foundation and data transparency

The solar project builds on my coauthored research:

1. **Díaz-Bedoya et al. (2023).** [Forecasting Univariate Solar Irradiance using Machine learning models: A case study of two Andean Cities](https://doi.org/10.1016/j.enconman.2023.117618). *Energy Conversion and Management*, 296, 117618.
2. **Díaz-Bedoya et al. (2025).** [Solar Irradiance Forecasting With Deep Learning and Ensemble Models: LSTM, Random Forest and Extra Trees With Multivariate Meteorological Data](https://doi.org/10.1049/stg2.70019). *IET Smart Grid*, 8, e70019.

Cuenca was the primary dataset in the earlier research, with Quito supporting it. This portfolio isolates Quito and uses a different forecast horizon, daylight window and evaluation protocol. Its model rankings are new portfolio results, not a replication or replacement of the papers’ conclusions. The Tail Risk Lab is a separate educational experiment, not a published result or evidence that Quito irradiance follows a power law.

The solar audit found no missing values, missing hourly timestamps or duplicate timestamps. The pipeline stops if these arise rather than silently imputing or deleting data. REMMAQ attribution is supported by the first article, but the exact CSV export, station, units, timezone, aggregation and reuse license still need confirmation. Published measurement context is recorded separately from verified export metadata; the project owner confirmed permission to publish the solar data on 2026-10-18. This confirmation does not establish a general downstream reuse license. See [data provenance](data_source.json), [research context](research.json) and the [detailed solar data notice](SOLAR_FORECASTING.md#data-notice).

## Reproduce and inspect

Python 3.13 was used for the recorded runs. From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/train.py
.\.venv\Scripts\python.exe src/risk_benchmark.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

With Node.js installed, check the browser’s decision calculations:

```powershell
node tests/risk_engine.test.cjs
```

Training regenerates saved reports and dashboard data. Recorded versions, configurations and seeds support inspection; timings and floating-point results can vary by platform. Avoid repeatedly tuning against the published test results.

| Location | Contents |
|---|---|
| [Solar methodology](SOLAR_FORECASTING.md) | Forecast contract, architectures, leakage controls, validation limits and provenance |
| [Tail-risk methodology](RISK_LAB.md) | Generators, forecasting protocol, decision equations and interpretation |
| [Source](src/) | Training pipelines, recurrent models and optional local server |
| [Dashboards](docs/) | Offline solar presentation and separate `risk/` lab |
| [Results](results/) | Saved metrics, fold scores, forecasts and experiment reports |
| [Tests](tests/) | Pipeline, recurrent-model and decision-engine checks |

For an optional local server, double-click **`start-dashboard.cmd`**. The solar page opens at `http://127.0.0.1:8000`; the Tail Risk Lab is at `/risk/`. **Close the terminal or press Ctrl+C to stop the server.** No server is needed when opening the HTML files directly.

The `docs/` directory is prepared for GitHub Pages. The project owner confirmed permission to publish the solar data on 2026-10-18; see `data_source.json`. Source license documentation and export metadata remain unverified. This workspace has not been deployed.

### Deploy to GitHub Pages

1. Confirm permission to publish the solar observations, including those embedded in `docs/data.js` and saved reports, before uploading the repository or publishing the site.
2. Run the checks above using the project's Python environment, and verify both dashboards locally.
3. Commit the project and push it to the chosen GitHub repository. Local environments and caches are excluded by `.gitignore`.
4. In the repository's **Settings → Pages**, choose **Deploy from a branch**, select the branch containing the site (normally `main`) and **`/docs`**, then save.
5. Wait for the Pages deployment to finish and use the published URL shown in Settings. For a project repository, the solar dashboard is at `https://USERNAME.github.io/REPOSITORY/` and the Tail Risk Lab is at `https://USERNAME.github.io/REPOSITORY/risk/`.
6. Verify both live dashboards, then add their actual URLs to this README.

The site needs no Python server or model training during deployment. The `docs/.nojekyll` marker disables Jekyll processing. Future pushes to the publishing branch update the website; include regenerated dashboard data when publishing new model results.
