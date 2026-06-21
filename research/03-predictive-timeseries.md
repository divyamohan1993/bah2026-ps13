# Phase 3a — Predictive Fault Analytics: Time-Series Forecasting & Regression Methods

**Problem Statement 13 — Air-Gapped Predictive Copilot for Secure MPLS Operations**
**Domain: Forecasting / Regression half of the analytics engine (target 15–18 distinct methods)**
**Author: deep-research specialist (forecasting). Sibling agent covers anomaly-detection / graph / change-point methods and continues the master numbering.**
**Date: 2026-06-20**

---

## 0. Purpose & framing

The platform must answer **Q1: "What is likely to fail next — and when?"** Forecasting is the engine behind the *"and when"*. Threshold alerts are reactive; the win condition (35% of score = "prediction accuracy and lead time") is **forecasting degradation precursors with enough LEAD TIME to intervene** and estimating **TIME-TO-IMPACT** (when a metric crosses an SLA/threshold).

Two complementary jobs:

1. **Trajectory forecasting** — predict the future curve of each metric (interface utilization, latency, jitter, packet-loss %, BGP update churn, tunnel rekey interval) over a horizon H (e.g. next 5–60 min). From the forecast trajectory + its uncertainty band we read off **time-to-impact = first time the predicted band crosses a threshold**.
2. **Time-to-event / survival** — directly model "time until SLA breach / link saturation" as a hazard, which yields a calibrated lead-time distribution and a risk score even when the precursor is not a smooth trend.

Everything below is selected for **fully offline / air-gapped** operation: pure-Python or local-binary libraries, model weights downloadable once and bundled, **CPU-runnable** wherever possible (the demo box may have no GPU). Cloud-only services (e.g. Nixtla **TimeGPT** API) are explicitly flagged as **NOT air-gap compatible** and excluded from the core.

**Telemetry context (from idea.md):** SNMP interface util/latency/jitter/error counters; syslog + BGP/OSPF events; NetFlow/IPFIX flow + tunnel stats; SD-WAN controller streaming telemetry. Sampling is typically 10 s–60 s → high-frequency, regular, multivariate, many parallel series (per-interface, per-tunnel, per-site). This favors **global multi-series models** and **online/incremental updates**.

---

## 1. Master list of FORECASTING / REGRESSION methods (M1–M24)

> Numbered so the **sibling anomaly/graph/change-point agent can continue from M25 toward 30+**. These 24 are the forecasting/regression contributions; "≥30 distinct methods across the engine" is comfortably met once the sibling's anomaly methods (M25+) are appended.

### Family A — Classical statistical (univariate)
- **M1. ARIMA / SARIMA / SARIMAX** — autoregressive integrated moving average; SARIMA adds seasonality, SARIMAX adds exogenous regressors (e.g. time-of-day, scheduled-backup flag).
- **M2. AutoARIMA** — automatic (p,d,q)(P,D,Q) order selection via stepwise AICc search.
- **M3. Exponential Smoothing / Holt-Winters / ETS** — level+trend+seasonal smoothing; ETS = error-trend-season statespace family with automatic model selection.
- **M4. Theta method (and OptimizedTheta / DynamicTheta)** — decompose into "theta lines", forecast each; M3-competition winner, extremely strong simple baseline.
- **M5. (T)BATS** — trigonometric seasonality, Box-Cox, ARMA errors, multiple/complex seasonality (e.g. daily + weekly traffic cycles).
- **M6. Croston / SBA / TSB / ADIDA / IMAPA** — intermittent-demand methods for series that are mostly zero with rare spikes — maps directly onto **packet-loss counts, CRC/error counters, rekey-failure events, syslog-event rates**.

### Family B — Multivariate / state-space
- **M7. VAR / VARMA / VARMAX** — vector autoregression; jointly models correlated metrics (latency↔jitter↔loss↔util) and captures cross-series lead/lag (e.g. util rises → latency rises 30 s later).
- **M8. Kalman filter / linear Gaussian state-space (UnobservedComponents / structural time series)** — recursive level/trend/seasonal estimation with **O(1) online update per step**; native handling of missing samples and noisy SNMP polls; gives smoothed state + forecast + variance for free.
- **M9. Dynamic Factor Models (DFM)** — compress many interfaces/tunnels into a few latent factors and forecast the factors (good when hundreds of correlated series share common congestion drivers).

### Family C — Decomposition (+ model on components/residuals)
- **M10. STL decomposition** — seasonal-trend decomposition via LOESS; forecast deseasonalized series with any base model (STLForecast). Trend slope itself is a precursor signal.
- **M11. MSTL** — multiple-seasonality STL (daily + weekly + business-hour cycles simultaneously); faster and often more accurate than Prophet/NeuralProphet on multi-seasonal load.
- **M12. Prophet** — additive trend + multi-seasonality + holiday/changepoint model; robust to gaps/outliers, interpretable components, auto changepoint detection (changepoints = candidate regime shifts).
- **M13. NeuralProphet** — Prophet reformulated in PyTorch with AR-Net autoregression + optional covariates; better short-term dynamics, supports **conformal prediction intervals** natively.

### Family D — Gradient-boosted regression on lag/rolling features (global multi-series)
- **M14. LightGBM** (on lag + rolling-mean/std/min/max/quantile + calendar features) — fast, accurate, the workhorse global model; trains one model across ALL interfaces/tunnels.
- **M15. XGBoost** — same feature paradigm, strong regularization; good for native quantile objective (lead-time bounds).
- **M16. CatBoost** — handles categorical features (device-role, site, vpn-id, interface-type) natively without one-hot; robust defaults.

### Family E — Deep learning
- **M17. RNN family: LSTM / GRU / vanilla-RNN** — sequence models for nonlinear long-range temporal patterns; LSTM explicitly named in the problem statement and the de-facto network-traffic baseline.
- **M18. 1D-CNN / TCN (Temporal Convolutional Network) / BiTCN** — dilated causal convolutions; capture multi-scale bursts, parallelizable, cheaper than RNNs.
- **M19. DeepAR** — autoregressive RNN that outputs a **probability distribution** (not a point); trains globally across many series, gives sampled trajectories → directly yields time-to-impact distribution.
- **M20. N-BEATS / N-BEATSx / N-HiTS** — pure deep MLP-with-basis stacks for interpretable trend/seasonality; N-HiTS adds hierarchical multi-rate sampling for efficient **long-horizon** forecasts.
- **M21. Transformers for TS: Temporal Fusion Transformer (TFT), PatchTST, Informer, Autoformer, FEDformer, iTransformer** — attention-based; TFT gives variable-importance + quantile outputs (interpretability for the copilot); PatchTST is a top long-horizon univariate model; Informer/Autoformer target very long horizons.
- **M22. TimesNet / TSMixer / DLinear / NLinear / TiDE** — modern efficient architectures; **DLinear/NLinear are 1-layer linear models** that rival transformers on many benchmarks at tiny cost (excellent CPU-friendly deep baselines); TSMixer is a strong all-MLP multivariate model.

### Family F — Probabilistic / quantile forecasting (confidence + lead-time bounds)
- **M23. Quantile regression** (Gradient Boosting Quantile, LightGBM/XGBoost quantile objective, quantile NN heads) + **Conformal prediction** (split conformal, **CQR**, **EnbPI**, **ACI / AgACI**, Conformal PID) for *distribution-free, calibrated* intervals around ANY base forecaster.
  - *Note:* probabilistic outputs are also produced natively by M8 (Kalman variance), M12/M13 (Prophet intervals), M19 (DeepAR), M20/M21 (quantile heads). M23 is the **cross-cutting wrapper** that guarantees coverage and turns any point forecaster into a lead-time-bounded one.

### Family G — Time-to-event / survival ("time-to-impact" directly)
- **M24. Survival models: Cox Proportional Hazards, Random Survival Forest (RSF), Gradient-Boosted survival, DeepSurv / DeepHit, Accelerated Failure Time (Weibull/log-normal AFT)** + **simple threshold-crossing extrapolation** baseline (fit local linear/Theil-Sen slope to recent forecast trajectory → extrapolate to threshold → time-to-impact with CI). Survival models consume the engineered features and the injected-fault ground-truth labels to output a **risk score + median time-to-breach + survival curve**.

> **Online/incremental forecasting is a *mode*, not a separate family** — it is satisfied by M8 (Kalman), M23-style EWMA extrapolation, online-ARIMA, and **River** regressors (Hoeffding-tree/linear/PA online learners). These provide **O(1) per-sample updates** so the engine adapts live to the streaming telemetry without retraining — called out per-method in §3 and §4.

**Foundation / zero-shot models (open-weight, air-gap-runnable) — see §5:** Amazon **Chronos / Chronos-Bolt / Chronos-2**, Google **TimesFM**, Salesforce **Moirai / Moirai-2**, **Lag-Llama**, **TimesFM-derived** community weights. These are *additional* forecasters (could be tagged M-FM1..M-FMn) used zero-shot when a series has little history.

---

## 2. Detailed method catalogue (description · when it wins · complexity/latency · best FREE offline library · air-gap suitability)

### Family A — Classical statistical

| # | Method | Short description | When it wins | Complexity / latency / online? | Best free offline library (exact package) | Air-gap |
|---|--------|-------------------|--------------|-------------------------------|-------------------------------------------|---------|
| M1 | **ARIMA/SARIMA/SARIMAX** | AR + differencing + MA, optional seasonal & exog terms | Stationary-after-differencing single metrics with clear short autocorrelation; need exog drivers (SARIMAX) | Fit O(n·iters); refit periodic. SARIMAX supports `append`/recursive update for cheap online extension | `statsmodels` (`SARIMAX`), `statsforecast` (`ARIMA`), `pmdarima` | ✅ pure Python |
| M2 | **AutoARIMA** | Auto order search (stepwise AICc) | You want a hands-off strong baseline per series; many series to fit | `statsforecast.AutoARIMA` is ~20× faster than `pmdarima`, Rust/Numba-compiled | **`statsforecast`** (preferred), `pmdarima` | ✅ |
| M3 | **ETS / Holt-Winters** | Exponential smoothing statespace, auto error/trend/season | Smooth trended/seasonal load; fast, robust, great default | Very fast; recursive by construction (each new point → O(1) state update) | **`statsforecast.AutoETS`** (4× faster than statsmodels), `statsmodels.ETSModel` | ✅ |
| M4 | **Theta** | Decompose into theta-lines, recombine | Strong universal baseline, near-SOTA-for-simplicity; few data points | Very fast | **`statsforecast`** (`AutoTheta`, `Theta`, `OptimizedTheta`, `DynamicTheta`) | ✅ |
| M5 | **(T)BATS** | Box-Cox + ARMA errors + trig multi-seasonal | Multiple/long/complex seasonality (daily+weekly) in one univariate model | Slower to fit | **`statsforecast`** (`AutoTBATS`), `tbats` | ✅ |
| M6 | **Croston / SBA / TSB / ADIDA / IMAPA** | Separate size & inter-arrival estimation for sparse series | **Packet-loss counts, error/CRC counters, rekey failures, syslog event rates** — mostly-zero spiky series where ARIMA fails | Very fast; recursive updates | **`statsforecast`** (`CrostonClassic`, `CrostonSBA`, `TSB`, `ADIDA`, `IMAPA`) | ✅ |

Sources: [Nixtla statsforecast (GitHub)](https://github.com/Nixtla/statsforecast), [statsforecast auto model selection](https://www.nixtla.io/blog/statsforecast-automatic-model-selection), [statsmodels state-space forecasting](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html), [Croston/TSB/ADIDA overview](https://www.prognostica.de/en/intermittent-demand-forecasting-methods/), [TSB extension paper](https://arxiv.org/html/2511.12749v1).

### Family B — Multivariate / state-space

| # | Method | Short description | When it wins | Complexity / online? | Library | Air-gap |
|---|--------|-------------------|--------------|----------------------|---------|---------|
| M7 | **VAR / VARMA / VARMAX** | Vector autoregression; each series regressed on lags of all series | **Cross-metric lead/lag**: util→latency→loss coupling; coordinated multi-interface congestion | Fit O(k²·p); forecast cheap | `statsmodels` (`VAR`, `VARMAX`) | ✅ pure Python |
| M8 | **Kalman / linear Gaussian state-space (Unobserved Components / structural TS)** | Recursive Bayesian level/trend/seasonal estimation | Noisy SNMP polls, **missing samples**, need **O(1) streaming update** + variance band for free; smooth drift detection | **O(1) per step** (true online); ideal for live telemetry | `statsmodels` (`UnobservedComponents`, `KalmanFilter`, `MLEModel`), `pykalman`, `filterpy` | ✅ |
| M9 | **Dynamic Factor Model (DFM)** | Latent common factors drive many series | Hundreds of correlated interfaces share common congestion driver — forecast few factors instead | Fit moderate | `statsmodels` (`DynamicFactor`, `DynamicFactorMQ`) | ✅ |

Sources: [statsmodels state-space methods](https://www.statsmodels.org/stable/statespace.html), [VARMA review](https://arxiv.org/pdf/2406.19702), [Kalman/state-space tutorial (Fulton)](http://www.chadfulton.com/fulton_statsmodels_2017/sections/2-state_space_models.html).

### Family C — Decomposition

| # | Method | Short description | When it wins | Complexity / online? | Library | Air-gap |
|---|--------|-------------------|--------------|----------------------|---------|---------|
| M10 | **STL (+ STLForecast)** | LOESS seasonal-trend split; model the remainder/trend | Single strong seasonality; want explicit **trend-slope precursor** + clean residual for anomaly hand-off | Fast | `statsmodels` (`STL`, `STLForecast`), `statsforecast` | ✅ |
| M11 | **MSTL** | Multiple seasonalities via iterated STL | **Daily+weekly+business-hour** load patterns together; beats Prophet/NeuralProphet on multi-seasonal load in speed & accuracy | Fast | **`statsforecast.MSTL`**, `statsmodels.MSTL` | ✅ |
| M12 | **Prophet** | Additive trend+seasonality+holidays+changepoints | Robust to gaps/outliers, interpretable components, **auto changepoints** = regime-shift candidates; good ops baseline | Moderate; refit periodic | `prophet` (formerly fbprophet; Stan backend, bundles offline) | ✅ (Stan compiled locally) |
| M13 | **NeuralProphet** | Prophet + AR-Net (PyTorch) + covariates + conformal | Need short-term autoregressive dynamics + native **conformal intervals**; richer than Prophet | Moderate (PyTorch CPU OK) | `neuralprophet` | ✅ (PyTorch local) |

Sources: [MSTL paper](https://arxiv.org/pdf/2107.13462), [NeuralProphet paper](https://arxiv.org/pdf/2111.15397), [MSTL vs Prophet/NeuralProphet (Nixtla electricity tutorial)](https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/electricityloadforecasting.html), [conformal + NeuralProphet](https://valeman.medium.com/probabilistic-forecasting-with-conformal-prediction-and-neuralprophet-af9c87901d94).

### Family D — Gradient-boosted regression on engineered features (GLOBAL multi-series) ⭐ project workhorse

| # | Method | Short description | When it wins | Complexity / online? | Library | Air-gap |
|---|--------|-------------------|--------------|----------------------|---------|---------|
| M14 | **LightGBM** | Boosted trees on lag/rolling/calendar features, one global model across all series | **Default recommended core**: fast, accurate, scales to all interfaces/tunnels, handles many exogenous signals, native quantile loss for bounds | Fast train, ms inference; not natively online but cheap periodic refit | **`mlforecast`** (Nixtla, automates lag/rolling features + CV) wrapping `lightgbm`; or `skforecast` | ✅ |
| M15 | **XGBoost** | Same paradigm, strong regularization | Robust global model; `reg:quantile` objective for pinball/quantile bounds | Fast | `mlforecast`/`skforecast` + `xgboost` | ✅ |
| M16 | **CatBoost** | Boosting with native categorical handling | Many categoricals (site, device-role, vpn-id, interface-type) without one-hot; strong defaults | Fast | `mlforecast`/`skforecast` + `catboost` | ✅ |

Sources: [mlforecast (GitHub)](https://github.com/Nixtla/mlforecast), [automated feature engineering with mlforecast](https://www.nixtla.io/blog/automated-time-series-feature-engineering-with-mlforecast), [skforecast XGBoost/LightGBM guide](https://skforecast.org/0.14.0/user_guides/forecasting-xgboost-lightgbm.html), [multi-series LightGBM (Forecastegy)](https://forecastegy.com/posts/multiple-time-series-forecasting-with-lightgbm-in-python/).

### Family E — Deep learning

| # | Method | Short description | When it wins | Complexity / online? | Library | Air-gap |
|---|--------|-------------------|--------------|----------------------|---------|---------|
| M17 | **LSTM / GRU / RNN** | Recurrent nets for nonlinear long-range temporal patterns | Named in PS; de-facto traffic baseline; nonlinear bursts; multivariate w/ covariates | Train GPU-pref but CPU-OK for small nets; **online fine-tune possible** | **`neuralforecast`** (`LSTM`,`GRU`,`RNN`), `darts`, `pytorch-forecasting` | ✅ (PyTorch local) |
| M18 | **1D-CNN / TCN / BiTCN** | Dilated causal convolutions, multi-scale | Bursty congestion, parallel training, cheaper than RNN | Fast inference | **`neuralforecast`** (`TCN`,`BiTCN`), `darts` (`TCNModel`) | ✅ |
| M19 | **DeepAR** | Autoregressive RNN → probability distribution | **Probabilistic** global forecasts; sampled trajectories → time-to-impact distribution; handles many related series | Train moderate; sampling at inference | **`neuralforecast`** (`DeepAR`), `gluonts` (`DeepAREstimator`), `darts` | ✅ |
| M20 | **N-BEATS / N-HiTS** | Deep basis-expansion MLP stacks | Interpretable trend/seasonal; **N-HiTS for efficient long-horizon**; strong pure-DL baseline | N-HiTS very efficient | **`neuralforecast`** (`NBEATS`,`NBEATSx`,`NHITS`), `darts` | ✅ |
| M21 | **TFT / PatchTST / Informer / Autoformer / iTransformer** | Attention-based seq2seq | **TFT** = variable importance + quantiles (copilot interpretability); **PatchTST** top long-horizon univariate; Informer/Autoformer very-long-horizon | Train GPU-pref; inference CPU-OK | **`neuralforecast`** (`TFT`,`PatchTST`,`Informer`,`Autoformer`,`FEDformer`,`iTransformer`), `pytorch-forecasting` (`TFT`), `darts` | ✅ |
| M22 | **TimesNet / TSMixer / DLinear / NLinear / TiDE** | Modern efficient (mostly non-attention) | **DLinear/NLinear = 1-layer, near-SOTA at tiny CPU cost** (ideal air-gap DL baseline); TSMixer strong multivariate; TiDE efficient | Very fast (linear) → moderate | **`neuralforecast`** (`DLinear`,`NLinear`,`TimesNet`,`TSMixer`,`TiDE`), `darts` (`DLinearModel`,`NLinearModel`) | ✅ |

Sources: [neuralforecast model list (README)](https://github.com/Nixtla/neuralforecast/blob/main/README.md), [PatchTST best univariate / TSMixer best multivariate](https://aihorizonforecast.substack.com/p/tsmixer-googles-innovative-deep-learning), [DLinear-PatchTST ensemble](https://www.researchgate.net/publication/391134489_Enhancing_Long-Term_Time_Series_Forecasting_via_Hybrid_DLinear-PatchTST_Ensemble_Framework), [darts models](https://github.com/unit8co/darts), [DeepAR paper](https://arxiv.org/pdf/1704.04110).

### Family F — Probabilistic / quantile (confidence + lead-time bounds) ⭐ essential for time-to-impact

| # | Method | Short description | When it wins | Complexity / online? | Library | Air-gap |
|---|--------|-------------------|--------------|----------------------|---------|---------|
| M23 | **Quantile regression + Conformal prediction (split / CQR / EnbPI / ACI / AgACI / Conformal-PID)** | Distribution-free calibrated prediction intervals around ANY base forecaster; quantile objective learns p10/p50/p90 directly | You need **trustworthy uncertainty bands** to compute time-to-impact as "first time the upper band crosses threshold" and to give the copilot a confidence score with **coverage guarantees** | Split conformal: O(1) calibration; **EnbPI/ACI update online** (no refit), adapt to drift | **`MAPIE`** (split, CQR, time-series), **`mapie`/`puncc`/`crepes`** for EnbPI/ACI; quantile loss in `lightgbm`/`xgboost`/`neuralforecast` | ✅ |

Sources: [MAPIE (GitHub)](https://github.com/scikit-learn-contrib/MAPIE), [MAPIE docs](https://mapie.readthedocs.io/en/latest/), [Adaptive Conformal Inference (Zaffran 2022)](https://proceedings.mlr.press/v162/zaffran22a/zaffran22a.pdf), [EnbPI (Xu & Xie)](https://arxiv.org/pdf/2010.09107), [Conformal PID control](https://arxiv.org/pdf/2307.16895), [gentle intro to conformal TS](https://arxiv.org/pdf/2511.13608).

### Family G — Time-to-event / survival ("time-to-impact" directly) ⭐ differentiator

| # | Method | Short description | When it wins | Complexity / online? | Library | Air-gap |
|---|--------|-------------------|--------------|----------------------|---------|---------|
| M24 | **Cox PH · Random Survival Forest · Gradient-Boosted Survival · DeepSurv/DeepHit · AFT (Weibull/log-normal)** + **threshold-crossing extrapolation baseline** | Model "time until SLA breach / saturation / flap" as a hazard from engineered features + fault labels → risk score, median time-to-impact, full survival curve. Extrapolation baseline: Theil-Sen/linear slope on recent trajectory → solve for threshold-crossing time + CI | When the failure is event-like (rekey storm, flap onset) and not a smooth trend; gives a **calibrated lead-time** and risk ranking the copilot can verbalize ("70% chance of breach within 12 min") | Train moderate; inference fast; extrapolation O(1) online | **`scikit-survival`** (`CoxPHSurvivalAnalysis`,`RandomSurvivalForest`,`GradientBoostingSurvivalAnalysis`), **`lifelines`** (Cox, Weibull/LogNormal AFT, Kaplan-Meier), **`pycox`** (DeepSurv/DeepHit/Cox-Time), `scipy.stats.theilslopes` for extrapolation | ✅ |

Sources: [scikit-survival (JMLR)](https://jmlr.org/papers/volume21/20-729/20-729.pdf), [lifelines vs scikit-survival vs pycox tour](https://sites.google.com/view/survival-analysis-tutorial), [RSF implementation](https://ujangriswanto08.medium.com/step-by-step-implementation-of-random-survival-forest-in-python-or-r-b27f4cd86a0d), [auton-survival](https://arxiv.org/pdf/2204.07276).

### Online / incremental mode (O(1) updates) — spans M8, M23, and River

| Method | Description | O(1)? | Library | Air-gap |
|--------|-------------|-------|---------|---------|
| **Kalman / state-space (M8)** | recursive predict-update | ✅ true O(1)/step | `statsmodels`, `filterpy` | ✅ |
| **EWMA / EWMM extrapolation** | exponentially weighted level/trend; IMA(1,1) ≈ SES; extrapolate to threshold | ✅ O(1)/step, tiny memory | `pandas.DataFrame.ewm`, custom; theory in EWMM paper | ✅ |
| **Online ARIMA / SARIMAX append** | extend fitted model with new obs without full refit | ✅ near-O(1) | `statsmodels` `.append()/.extend()` | ✅ |
| **River online regressors** | Hoeffding-tree / linear / Passive-Aggressive / SGD / KNN, learn one row at a time; drift detectors (ADWIN) | ✅ instance-incremental, O(1) update, Cython VectorDict | **`river`** | ✅ |

Sources: [River (GitHub)](https://github.com/online-ml/river), [River (JMLR)](https://www.jmlr.org/papers/volume22/20-1380/20-1380.pdf), [EWMM paper](https://arxiv.org/pdf/2404.08136), [adaptive LSTM + online learning beats ARIMA for traffic](https://dl.acm.org/doi/10.1145/3703447).

---

## 3. Foundation / zero-shot time-series models (open-weight, air-gap)

These give **strong forecasts with zero training**, useful for newly-provisioned interfaces/tunnels that lack history, and as diverse ensemble members. **Critical air-gap rule:** download weights from Hugging Face **once on a connected machine**, bundle into the offline image, then load via a **local path** (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`). No runtime network needed.

| Model | Open weights? | Local package | CPU-runnable? | Zero-shot | Notes / air-gap |
|-------|---------------|---------------|---------------|-----------|-----------------|
| **Amazon Chronos / Chronos-Bolt / Chronos-2** | ✅ Apache-2.0 | `chronos-forecasting`; also via `autogluon.timeseries` | ✅ **Bolt variants run on CPU**, up to ~250× faster & 20× more memory-efficient than original Chronos | ✅ excellent | **Best air-gap pick.** HF: `amazon/chronos-bolt-{tiny,small,base}`, `amazon/chronos-2`. `model_path` accepts a **local directory** → fully offline. Sizes 9M–710M. |
| **Google TimesFM (1.0 / 2.0 / 2.5)** | ✅ | `timesfm` (`pip install timesfm[torch]`) | ✅ (200M decoder, CPU feasible) | ✅ strong | HF: `google/timesfm-2.0-500m-pytorch`, `google/timesfm-2.5-200m-pytorch`; `from_pretrained` can read pre-downloaded weights offline. |
| **Salesforce Moirai / Moirai-2** | ✅ | `uni2ts` | ✅ small variants | ✅ universal, **native multivariate + covariates + irregular sampling** (good for mixed-rate IoT/telemetry) | HF: `Salesforce/moirai-2.0-R-small` etc. Download once, load locally. |
| **Lag-Llama** | ✅ permissive | `lag-llama` (+ `gluonts`) | ✅ **CPU or GPU** | few-/zero-shot | Probabilistic decoder-only; checkpoint downloadable for offline use. |
| **Nixtla TimeGPT** | ❌ **CLOUD API ONLY** | `nixtla` SDK → api.nixtla.io | n/a | ✅ | 🚫 **NOT air-gap compatible — explicitly excluded.** Listed only to warn the team. |

Sources: [2026 TS foundation-model toolkit](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/), [Chronos-Bolt runs on CPU + local model_path (AutoGluon docs)](https://auto.gluon.ai/stable/_modules/autogluon/timeseries/models/chronos/model.html), [Chronos-Bolt on Hugging Face](https://huggingface.co/amazon/chronos-bolt-base), [Chronos-Bolt + AutoGluon (AWS)](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/), [TimesFM install/from_pretrained](https://pypi.org/project/timesfm/), [TimesFM 2.5 on HF](https://huggingface.co/google/timesfm-2.5-200m-pytorch), [deploy Chronos/Moirai/TimesFM/Lag-Llama](https://www.spheron.network/blog/deploy-time-series-foundation-models-gpu-cloud/).

> **Recommendation:** include **Chronos-Bolt-base** (CPU, Apache-2.0, local path) as the zero-shot ensemble member and cold-start forecaster. Optionally add **TimesFM-2.x** for diversity. Keep them OPTIONAL/feature-flagged so the engine degrades gracefully on a tiny demo box.

---

## 4. Ensembling & stacking strategy (robust, cross-verified)

The PS demands "robust, cross-verified findings" and lists "ensemble classifiers". Combining many diverse forecasters reduces variance and raises confidence; **cross-model agreement is itself a signal** (high agreement → high confidence; disagreement → flag uncertainty / widen the lead-time band → fill gaps). The equal-weight mean is a famously hard-to-beat, robust baseline; learned stacking adds accuracy.

Strategies (in increasing sophistication):

1. **Simple / trimmed mean & median ensemble** — average point forecasts (median = robust to a single blown-up model). Strong, near-free baseline.
2. **Inverse-error / weighted averaging** — weight each model by recent rolling backtest accuracy (e.g. inverse-MASE on the last window); models that have been good lately count more. Cheap to update online.
3. **Best-on-window selection** — pick, per series and per step, the model with lowest error on the trailing window. Good when regimes switch (different scenario → different best model).
4. **Stacking / meta-learner** — train a meta-regressor (LightGBM or linear) on the base models' out-of-fold predictions + features; learns context-dependent combination. **Multi-layer stacking** lets later layers correct earlier errors (current SOTA in benchmarks).
5. **Probabilistic ensemble / quantile averaging** — average quantiles or pool DeepAR/conformal intervals across members → calibrated ensemble band; **conformalize the ensemble** (MAPIE on top) for coverage guarantees.
6. **Agreement-as-confidence** — compute cross-model spread (std/IQR of member forecasts) at each horizon; feed to the copilot as a confidence score and to alert-prioritization (Objective 4). Disagreement = "need human review".

**Off-the-shelf, fully-offline meta-frameworks:**
- **`AutoGluon-TimeSeries`** — auto-trains statistical + ML + DL + Chronos models and **builds a weighted ensemble automatically**; runs offline (point models) with local Chronos weights. Great for a strong baseline fast.
- **`darts`** — `RegressionEnsembleModel` (learned weights) and `NaiveEnsembleModel` (averaging) across any darts models + built-in backtesting.
- **`StatsForecast`/`MLForecast`/`NeuralForecast`** share a tidy API → easy to align outputs and hand-roll the meta-learner.

Sources: [Multi-layer stack ensembles for TS (Amazon)](https://arxiv.org/pdf/2511.15350), [AutoGluon TS ensembles](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-ensembles.html), [equal-weight mean is a robust baseline / combinations beat singles](https://arxiv.org/pdf/2108.08723), [cost of ensembling](https://arxiv.org/pdf/2506.04677), [darts ensembling](https://github.com/unit8co/darts).

---

## 5. Evaluation plan (accuracy AND lead time)

Score on **two axes**: (a) how accurate is the forecast, (b) how early & reliably does it warn. Phase 3 explicitly evaluates *precision, recall, false-positive rate, and prediction lead time*.

### A. Point & probabilistic accuracy
- **MAE, RMSE** — scale-dependent; RMSE penalizes large misses (saturation spikes) harder.
- **MASE** — scale-free, comparable across interfaces/metrics; ratio vs seasonal-naive; **primary cross-series metric**.
- **sMAPE** — scale-free percentage, symmetric; report but beware near-zero values (loss %, jitter).
- **Pinball (quantile) loss** — evaluates each predicted quantile; **the metric for lead-time bounds**.
- **CRPS** — full-distribution score for probabilistic models (DeepAR, conformal, foundation models).
- **Interval coverage & width** — does the 90% band actually cover ~90%? (PICP / MIS) — validates conformal calibration.

### B. Early-warning / lead-time metrics (the differentiator — 35% of score)
For each injected fault with known onset/breach time t_breach:
- **Mean lead time** = t_breach − t_firstvalidalert (only counting alerts that precede impact). Headline KPI.
- **Time-to-detection (TTD)** — latency from precursor onset to alert.
- **Early-warning precision / recall / FPR** — treat "predicted breach within horizon H" as a binary detector vs ground-truth labels; sweep threshold → **precision-recall & ROC curves**, pick operating point minimizing alert fatigue (Objective 4).
- **Time-to-impact (TTI) error** — |predicted_TTI − actual_TTI| and calibration of the TTI distribution (does "12 min ± 3" hold?). For survival models: **concordance index (C-index)** + integrated Brier score / time-dependent AUC.
- **Lead-time vs precision trade-off curve** — earlier warnings usually cost precision; plot the frontier per scenario.

### C. Backtesting protocol (mandatory, no leakage)
- **Rolling-origin / walk-forward** with **expanding window** (preferred for shorter series) and a **sliding window** variant for drift robustness. Train→predict next H→advance origin→repeat. This is the only correct way to estimate live performance.
- Built-in support: `statsforecast`/`mlforecast`/`neuralforecast` `.cross_validation()`, `darts.historical_forecasts()`/`backtest()`, `skforecast` backtesting, `AutoGluon` evaluation, `MAPIE` time-series CV.
- **Gap/embargo** between train and test windows to avoid leakage from rolling features. Evaluate per-scenario AND aggregate.

Sources: [forecasting metrics (AutoGluon)](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html), [rolling-origin / walk-forward backtesting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/), [backtesting overview](https://milvus.io/ai-quick-reference/what-is-backtesting-in-time-series-forecasting), [MASE/sMAPE/pinball/CRPS explainer](https://eshban9492.medium.com/understanding-evaluation-metrics-for-time-series-forecasting-5c8a3c877654), [skforecast backtesting](https://skforecast.org/0.7.0/introduction-forecasting/introduction-forecasting).

---

## 6. Mapping methods → the 4 validation scenarios

| Scenario | Signal signature | Best forecasting methods | Time-to-impact mechanism |
|----------|------------------|--------------------------|--------------------------|
| **1. Progressive congestion buildup (hub-spoke link)** | Smooth monotonic ↑ in interface utilization, gradual latency drift, queue-depth growth; strong daily seasonality | **M14 LightGBM (global)** on lag+rolling+calendar; **M11 MSTL** / **M3 ETS** for trend+seasonality; **M8 Kalman trend** for live slope; **M22 DLinear** as cheap DL; **M12 Prophet** for interpretable trend/changepoints | Forecast util trajectory + **M23 conformal band**; **TTI = first time upper band crosses saturation threshold** (e.g. 85% util). Trend-slope extrapolation as fast O(1) check. |
| **2. BGP route-flap cascade** | Bursty spikes in BGP UPDATE/withdraw counts, adjacency-change events, transient latency/loss as paths reconverge; mostly-zero-then-burst | **M6 Croston/TSB** for sparse update-count spikes; **M7 VAR** to capture flap→downstream-reroute coupling across sites; **M17/M19 LSTM/DeepAR** on event-rate sequences; **M24 survival** ("time-to-next-flap-storm") on flap features | Survival hazard → risk of cascade within H; VAR/DeepAR forecast of update-rate crossing flap-storm threshold → lead time. (Detection of the flap *event* itself is the sibling's change-point/anomaly job; forecasting the *escalation* is ours.) |
| **3. Intermittent MPLS underlay / tunnel degradation** | Rising packet-loss %, jitter trend, periodic rekey, intermittent drops; nonstationary, regime-switching | **M6 Croston/TSB** for loss/error counters; **M11 MSTL**+**M8 Kalman** for jitter/latency trend under noise/missing data; **M19 DeepAR** probabilistic loss-progression; **M13 NeuralProphet** + conformal; **River online** to adapt to regime shifts live | Probabilistic forecast of loss%/jitter trajectory; **TTI = first time predicted loss crosses SLA (e.g. 1%)**; survival model on tunnel-health features → median time-to-degradation. |
| **4. Controller misconfig → policy drift** | Step/level-shift in QoS class-mappings, route-policy counters, flow-distribution; slow drift then divergence vs expected baseline | **M12 Prophet / M10 STL** (changepoint + residual-vs-baseline); **M8 Kalman** level-shift detection; **M14 LightGBM** forecast vs realized divergence; **M23 conformal** residual band (out-of-band = drift) | Forecast the *expected* (pre-drift) trajectory; **TTI / drift-onset = sustained excursion of actuals beyond the conformal band**; extrapolate divergence to SLA-impact threshold. (Config-change *event* correlation = sibling/graph layer.) |

> Division of labour: **forecasting (this doc)** answers *"and when / time-to-impact / lead time"* via trajectory + bounds + survival. **Anomaly/change-point/graph (sibling)** answers *"is this abnormal right now / where in the topology"*. They feed each other: change-points reset/segment the forecasters; forecast residuals feed anomaly detectors.

---

## 7. Recommended CORE ensemble for the project

Optimized for **air-gap, CPU-first, strong lead time, interpretability, robustness** — a tiered set so we are never reliant on one model and cross-model agreement gives a confidence score:

**Tier 1 — always-on, cheap, online (run every sample):**
1. **M8 Kalman / structural state-space** (`statsmodels`) — O(1) live trend + variance band; missing-data robust.
2. **M3 AutoETS** + **M4 Theta** + **M2 AutoARIMA** (`statsforecast`) — fast classical baselines, auto per-series.
3. **M6 Croston/TSB** (`statsforecast`) — for sparse loss/error/event counters.
4. **River** online regressor + EWMA extrapolation — instant adaptation + O(1) threshold-crossing TTI.

**Tier 2 — global accuracy workhorse (periodic refit, minutes):**
5. **M14 LightGBM** global model via **`mlforecast`** (lag+rolling+calendar+exog) — the primary accuracy driver; **+ M15 XGBoost-quantile / M16 CatBoost** for diversity & bounds.
6. **M11 MSTL** + **M12 Prophet** (`statsforecast` / `prophet`) — multi-seasonality + interpretable changepoints for the copilot.

**Tier 3 — deep & probabilistic (where GPU/time allows; CPU-OK variants chosen):**
7. **M19 DeepAR** + **M22 DLinear/NLinear** + **M20 N-HiTS** (`neuralforecast`) — probabilistic + cheap-DL + long-horizon. Add **M21 TFT** for variable-importance explanations.
8. **Chronos-Bolt-base** (zero-shot, CPU, local weights) — cold-start + diversity.

**Cross-cutting wrappers:**
9. **M23 Conformal (MAPIE / EnbPI / ACI)** over the ensemble → calibrated lead-time bounds + confidence.
10. **M24 Survival (RSF / Cox, `scikit-survival`)** on engineered features → direct time-to-impact + risk ranking.
11. **Ensemble layer:** inverse-MASE weighted average + median, optional LightGBM stacker; **AutoGluon-TS** as a turnkey alternative; cross-model spread → confidence score feeding alert prioritization.

This spans **24 forecasting/regression methods** across **7 families + foundation models + online mode + ensembling + survival**, comfortably meeting the forecasting-half target (15–18) and, with the sibling's anomaly/graph/change-point methods (M25+), the engine's ≥30-method requirement.

---

## 8. Exact offline library shortlist (pip-installable, no runtime network)

| Need | Package(s) | Methods covered |
|------|------------|-----------------|
| Fast classical + intermittent + MSTL/Theta/TBATS | **`statsforecast`** | M1–M6, M11 |
| Mature state-space / VAR / Kalman / STL / DFM | **`statsmodels`** | M1, M7–M10 |
| AutoARIMA (alt) | `pmdarima` | M2 |
| Global ML + auto feature engineering + CV | **`mlforecast`** (+`lightgbm`,`xgboost`,`catboost`) | M14–M16 |
| Alt global ML + backtesting | `skforecast` | M14–M16 |
| Neural forecasting (RNN/TCN/DeepAR/NBEATS/NHITS/TFT/PatchTST/DLinear/TimesNet/…) | **`neuralforecast`** | M17–M22 |
| All-in-one TS (classical+ML+DL+ensemble+backtest+probabilistic) | **`darts`** | M1,M3,M4,M12,M17–M22, ensembles |
| TFT/DeepAR with rich covariates | `pytorch-forecasting` | M19, M21 |
| Probabilistic DeepAR / GluonTS models / Lag-Llama backend | `gluonts` | M19, foundation |
| Decomposition + AR-Net + conformal | `neuralprophet`, `prophet` | M12, M13 |
| Conformal / calibrated intervals | **`MAPIE`** (+ `puncc`/`crepes` for EnbPI/ACI) | M23 |
| Survival / time-to-event | **`scikit-survival`**, `lifelines`, `pycox` | M24 |
| Online / streaming O(1) | **`river`** | online mode |
| Zero-shot foundation (open weights, local) | `chronos-forecasting`, `timesfm`, `uni2ts` (Moirai), `lag-llama`; or via `autogluon.timeseries` | foundation |
| Turnkey ensemble/AutoML (offline w/ local weights) | **`autogluon.timeseries`** | ensembling |

**Air-gap install/runtime checklist:**
- Build a wheelhouse: `pip download` all packages + deps on a connected box → copy → `pip install --no-index --find-links=./wheelhouse ...` offline.
- Pre-download any HF weights (Chronos/TimesFM/Moirai/Lag-Llama) to a bundled directory; set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; load by **local path**.
- Prophet (Stan) and the boosting/torch libs compile/run locally — no network. Verify with the firewall closed.
- 🚫 Do **not** install/import `nixtla` (TimeGPT) in the runtime path — it calls a cloud API.

---

## 9. Source URLs (consolidated)

- statsforecast: https://github.com/Nixtla/statsforecast · https://www.nixtla.io/blog/statsforecast-automatic-model-selection
- statsmodels state-space/Kalman/VAR/STL: https://www.statsmodels.org/stable/statespace.html · https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html · http://www.chadfulton.com/fulton_statsmodels_2017/sections/2-state_space_models.html
- VARMA review: https://arxiv.org/pdf/2406.19702
- Intermittent demand (Croston/TSB/ADIDA): https://www.prognostica.de/en/intermittent-demand-forecasting-methods/ · https://arxiv.org/html/2511.12749v1
- MSTL: https://arxiv.org/pdf/2107.13462 · https://nixtlaverse.nixtla.io/statsforecast/docs/tutorials/electricityloadforecasting.html
- Prophet/NeuralProphet: https://arxiv.org/pdf/2111.15397 · https://valeman.medium.com/probabilistic-forecasting-with-conformal-prediction-and-neuralprophet-af9c87901d94
- mlforecast / skforecast / global ML: https://github.com/Nixtla/mlforecast · https://www.nixtla.io/blog/automated-time-series-feature-engineering-with-mlforecast · https://skforecast.org/0.14.0/user_guides/forecasting-xgboost-lightgbm.html · https://forecastegy.com/posts/multiple-time-series-forecasting-with-lightgbm-in-python/
- neuralforecast / deep models: https://github.com/Nixtla/neuralforecast/blob/main/README.md · https://aihorizonforecast.substack.com/p/tsmixer-googles-innovative-deep-learning · https://www.researchgate.net/publication/391134489_Enhancing_Long-Term_Time_Series_Forecasting_via_Hybrid_DLinear-PatchTST_Ensemble_Framework
- darts: https://github.com/unit8co/darts · https://unit8co.github.io/darts/
- DeepAR: https://arxiv.org/pdf/1704.04110
- Conformal prediction (MAPIE/EnbPI/ACI/PID): https://github.com/scikit-learn-contrib/MAPIE · https://mapie.readthedocs.io/en/latest/ · https://proceedings.mlr.press/v162/zaffran22a/zaffran22a.pdf · https://arxiv.org/pdf/2010.09107 · https://arxiv.org/pdf/2307.16895 · https://arxiv.org/pdf/2511.13608
- Survival analysis: https://jmlr.org/papers/volume21/20-729/20-729.pdf · https://sites.google.com/view/survival-analysis-tutorial · https://arxiv.org/pdf/2204.07276 · https://ujangriswanto08.medium.com/step-by-step-implementation-of-random-survival-forest-in-python-or-r-b27f4cd86a0d
- Online / streaming: https://github.com/online-ml/river · https://www.jmlr.org/papers/volume22/20-1380/20-1380.pdf · https://arxiv.org/pdf/2404.08136
- Foundation models (offline): https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/ · https://huggingface.co/amazon/chronos-bolt-base · https://auto.gluon.ai/stable/_modules/autogluon/timeseries/models/chronos/model.html · https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/ · https://pypi.org/project/timesfm/ · https://huggingface.co/google/timesfm-2.5-200m-pytorch · https://www.spheron.network/blog/deploy-time-series-foundation-models-gpu-cloud/
- Ensembling: https://arxiv.org/pdf/2511.15350 · https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-ensembles.html · https://arxiv.org/pdf/2108.08723 · https://arxiv.org/pdf/2506.04677
- Evaluation / backtesting: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html · https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/ · https://milvus.io/ai-quick-reference/what-is-backtesting-in-time-series-forecasting · https://eshban9492.medium.com/understanding-evaluation-metrics-for-time-series-forecasting-5c8a3c877654
- Network traffic forecasting / SLA / lead time: https://dl.acm.org/doi/10.1145/3703447 · https://arxiv.org/pdf/2309.03898 · https://towardsdatascience.com/from-reactive-to-predictive-forecasting-network-congestion-with-machine-learning-and-int/

---

*End of forecasting-half catalogue (M1–M24 + foundation models). Sibling agent: please continue the master numbering at **M25** for anomaly-detection / graph / change-point methods.*
