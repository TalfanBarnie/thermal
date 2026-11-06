# Thermal anomaly monitoring over Iceland from Meteosat — design specification

Status: draft v0.1, 2026-11-06. This document is the agreed design for the
`thermal` project. Items marked **[verify]** are assumptions about external
services or data that must be checked during the first milestone.

## 1. Goal

Detect and quantify volcanic thermal anomalies over Iceland, with the
Reykjanes peninsula as the first target, from geostationary imagery:

- Meteosat Second Generation (MSG) SEVIRI, both the 0° full disk service
  (15 min) and the Rapid Scan Service (RSS, 5 min, from 9.5°E).
- Meteosat Third Generation (MTG) FCI (10 min, from 0°).

The system has two parts:

1. A **backend** that runs in a GitHub Action: fetches region cutouts from the
   EUMETSAT Data Store, applies a pre-trained background model for the
   mid-infrared (MIR, 3.9 µm) channel, computes residual (anomaly) images and
   region statistics, and writes everything to a Backblaze B2 bucket.
2. A **single page web app** on GitHub Pages that reads from the bucket and
   shows the anomaly time series; clicking a point shows the images for that
   time.

A third, offline component is an **analysis package** that fits a Bayesian
effusion rate model to the anomaly time series of past eruptions (section 9).
It never runs in the Action.

### Non-goals for v1

- Cloud correction. The cloud mask is stored but not used.
- Plume scattering / vertical profiling. Residual images are stored so this
  can be done later.
- Low latency. Data Store latency (15 to 30 min) plus hourly scheduling is
  accepted. EUMETCast reception is a future input.

## 2. Core method

The MIR channel responds to sub-pixel hot sources far more strongly than the
thermal infrared (TIR) channels, so a model that predicts the MIR radiance of a
*non-volcanic* scene from the other channels and the geometry gives a
background estimate, and the observed-minus-predicted residual is the thermal
anomaly.

Earlier design iterations used a per-image "donut and hole" regression. That
has been replaced by **one model per satellite, trained once on archive
images from non-eruption periods, covering the whole of Iceland**, with
position and solar geometry as inputs. Consequences:

- No per-image training in the Action; inference only.
- Location-specific baselines (geothermal fields, terrain, coast) are learned.
- Twilight needs no special case: solar zenith angle is a continuous input.
- Any Icelandic volcano is monitored, not only Reykjanes.
- Correctness depends entirely on the eruption exclusion calendar (section 5).

Cloud is ignored in v1. Eruptions are often visible through cloud, and the
extremely oblique view projects emission scattered at height in the plume
onto the ground as a line of anomalies extending away from the sub-satellite
point. The monitoring region must contain that projection (section 4).

## 3. Data sources

All data comes from the EUMETSAT Data Store via the `eumdac` Python client,
using the **Data Tailor web service** to cut a region so that only a few MB
per timestamp are downloaded. The Data Tailor request must keep the **native
geostationary projection** (no reprojection), so that sub-pixel hot spots are
not smeared and the plume stack geometry is preserved. **[verify]** that the
Data Tailor accepts a region cut without a projection change for each
collection, and what the output netCDF contains (projection coordinates vs
lat/lon arrays).

| Source | Collection id **[verify]** | Cadence | Sub-satellite lon | Channels used |
|---|---|---|---|---|
| MSG 0° full disk, level 1.5 | `EO:EUM:DAT:MSG:HRSEVIRI` | 15 min | 0° | VIS0.6 VIS0.8 NIR1.6 IR3.9 WV6.2 WV7.3 IR8.7 IR9.7 IR10.8 IR12.0 |
| MSG RSS, level 1.5 | `EO:EUM:DAT:MSG:MSG15-RSS` | 5 min | 9.5° E | same |
| MSG cloud mask (0°) | `EO:EUM:DAT:MSG:CLM` | 15 min | 0° | cloud mask |
| MSG cloud mask (RSS) | `EO:EUM:DAT:MSG:RSS-CLM` | 5 min | 9.5° E | cloud mask |
| MTG FCI level 1c FDHSI | `EO:EUM:DAT:0662` | 10 min | 0° | all 16 |
| MTG FCI level 1c HRFI | `EO:EUM:DAT:0665` | 10 min | 0° | VIS0.6 NIR2.2 IR3.8 IR10.5 at high res |
| MTG FCI cloud mask | **[verify]** | 10 min | 0° | cloud mask |

Notes:

- The HRV channel is excluded (different grid, daytime only).
- Each satellite that occupied the 0° slot (Meteosat-8, 9, 10, 11) and each
  slot (0°, 9.5°E RSS) gets its own model. The *satellite key* is
  `<platform>-<slot>`, e.g. `msg11-0deg`, `msg10-rss`, `mtgi1-0deg`.
  **[verify]** which platform occupied which slot on which dates; the Data
  Store product metadata carries the platform name.
- Reykjanes eruptions 2021 to 2024 were imaged by Meteosat-11 at 0°
  **[verify]**, so the first model is `msg11-0deg`.
- Raw cutouts are kept permanently in the private bucket (section 9), because
  re-ordering from the Data Tailor is slow and quota-limited.

### Data Tailor usage

- One customisation per product (image) and per cloud mask product.
- Output format netCDF4, native projection, region of interest = the Iceland
  box in section 4. **[verify]** exact `eumdac.tailor_models.Chain` fields.
- The web service has concurrency and daily volume quotas **[verify]**; the
  client must poll, retry on failure, and respect a configurable concurrency
  limit. Backfill runs as a long-running batch over days, with resumable
  state, not as a single Action run.

## 4. Regions

All regions are defined in `config/regions.yaml` in WGS84 lat/lon and
applied to the native grid by point-in-polygon on pixel centres.

### Iceland box (fetch, training and inference region)

Bounding box of Iceland buffered by 20 km. Initial values, to be tuned:

| Edge | Value |
|---|---|
| North | 66.8° N |
| South | 63.1° N |
| West | 25.0° W |
| East | 13.0° W |

Every pixel in this box is a training sample (non-eruption images) and every
pixel gets a residual at inference. The 20 km buffer also covers the plume
stack projection: at ~70° viewing zenith the ground shift is ~2.7 km per km
of plume height, towards the sub-satellite point.

### Monitoring regions (time series summaries)

A monitoring region is a polygon over which residual statistics are
aggregated into one time series. Regions are per satellite slot because the
plume projection azimuth depends on the sub-satellite point (about SSE from
Reykjanes for 0°, further east for 9.5°E).

Initial regions:

- `reykjanes`: the peninsula west of about 21.5°W, extended ~30 km
  towards the sub-satellite point over the sea. To be tuned by inspecting
  residual images from a known eruption.
- Later: `grimsvotn`, `bardarbunga`, `askja`, `hekla`, `katla`, each a
  simple box with the same extension rule.

## 5. Eruption exclusion calendar

File: `data/eruptions.csv`, columns `volcano, start_date, end_date, source,
notes`. Initially scraped from Wikipedia (list of volcanic eruptions in
Iceland and the Reykjanes/Sundhnúkur eruption pages), then hand-curated and
committed. Must cover at least Eyjafjallajökull 2010, Grímsvötn 2011,
Holuhraun 2014–2015, Fagradalsfjall 2021, 2022, Litli-Hrútur 2023, and every
Sundhnúkur event from December 2023 onwards. The `end_date` of an ongoing
eruption is empty and treated as "today".

**Exclusion rule:** an image is excluded from training if its timestamp lies
in `[00:00 UTC on start_date, end_date + 30 days]` for any row. The 30 day
cooling tail is deliberately generous; the MIR anomaly of cooling lava decays
much faster, and the tail may be shortened once the cooling behaviour has
been examined.

The same file is the ground truth for validation (onset detection) and
defines the eruption windows for the effusion model.

## 6. Background model

### Samples

- Training images: for each satellite key, sample `N_train` (~1000)
  timestamps from the archive, stratified uniformly over calendar month and
  hour of day, excluding the calendar windows. Hold out whole days (not
  pixels) for validation: group k-fold by date.
- Each pixel in the Iceland box of each image is one sample. Pixels flagged
  invalid (space, missing line, saturated in any channel) are dropped.

### Features and target

| Feature | Note |
|---|---|
| All channels except IR3.9 and HRV | calibrated radiance; brightness temperature for IR channels as an additional derived input |
| Solar zenith angle | computed per pixel from a NOAA-style solar position routine in numpy |
| Relative azimuth (solar azimuth minus satellite azimuth) | captures forward-scatter and shadowing geometry |
| Latitude, longitude | location-specific baseline; viewing geometry is implied |
| Day of year (as sin, cos) | snow cover season |

Target: IR3.9 **radiance** (not brightness temperature), because the
anomaly in radiance units is additive and feeds the effusion model directly.
Inputs and target are standardised with statistics stored with the model.

### Candidate models

Both are trained on the identical sample set and compared:

1. Ridge regression on polynomial features (degree 2–3) — deterministic,
   fast, baseline.
2. Small multilayer perceptron (3–4 hidden layers, ~64–128 units, GELU)
   trained with PyTorch on CPU.

Both predict **mean and log-variance** (heteroscedastic Gaussian negative
log-likelihood loss for the MLP; for ridge, a second ridge fit on the
squared residuals), so inference yields a z-score as well as a residual.

### Evaluation

- Held-out residual histogram and standard deviation on non-eruption days,
  by solar zenith bin and by month, compared with the channel's noise
  equivalent temperature.
- Spatial residual maps averaged over held-out days (reveals learned or
  missed location effects).
- Detection check: residual time series over `reykjanes` for one known
  eruption, which is strictly out of training by construction.

### Model artefact

Exported to numpy `.npz` (weights, biases, activation name, feature list,
standardisation stats, metadata) so the Action's inference needs only numpy.
Stored at `models/<satellite_key>/<model_id>.npz` in the bucket with a
`latest.json` pointer. Training code may depend on torch and scikit-learn;
runtime code must not.

## 7. Anomaly products

Per timestamp and satellite key, computed in the Action:

- `residual`: observed minus predicted IR3.9 radiance, Iceland box, float32.
- `zscore`: residual divided by predicted standard deviation.
- `flags`: per pixel: saturated IR3.9, invalid, cloud (from the cloud mask
  product, stored but unused in v1).
- PNG renderings (fixed colour scales) of IR3.9 observed, IR3.9 predicted,
  residual, IR10.8, VIS0.6 (day only), cloud mask. Rendered in Python so
  the web app only displays images.

Per timestamp and monitoring region, appended to the time series:

| Field | Meaning |
|---|---|
| `time` | ISO 8601 UTC nominal image time |
| `sat` | satellite key |
| `region` | region name |
| `sum_pos` | sum of positive residual radiance over the region (primary statistic; integrates the plume stack) |
| `max` | maximum residual |
| `n_above` | number of pixels with z-score above a configured threshold |
| `vrp_w` | volcanic radiative power via the MIR method (Wooster et al. 2003): constant × pixel area × excess radiance, summed; unit conversion from SEVIRI radiance units **[verify]** |
| `n_sat` | saturated IR3.9 pixel count in region |
| `n_cloud` | cloudy pixel count in region (from cloud mask) |
| `n_valid` | valid pixel count |
| `sza_mean` | mean solar zenith angle over region |

Saturation of IR3.9 is tracked explicitly: SEVIRI saturates near 335 K, FCI
IR3.8 much higher **[verify]**, and the effusion model treats saturated
samples as upper-censored.

## 8. Backend pipeline and operations

### Package layout

```
thermal/                     Python package (runtime, numpy-only inference)
  config.py                  loads config/*.yaml, env vars, secrets
  datastore.py               eumdac search, Data Tailor customisation, download, retries
  geo.py                     native-grid geolocation (pyproj geos), pixel area, solar angles
  regions.py                 polygons -> pixel masks on a given grid
  features.py                feature matrix from a cutout (shared by training and inference)
  model.py                   load .npz, predict mean/variance
  anomaly.py                 residuals, flags, region statistics
  render.py                  PNG rendering
  storage.py                 B2 (S3 API via boto3): private and public buckets, state file, time series append
  pipeline.py                process one timestamp end to end; catch-up loop
  cli.py                     `thermal fetch|process|backfill|sample|train|evaluate`
training/                    torch/sklearn training and evaluation (separate extra)
analysis/                    effusion rate model (Stan), notebooks, separate extra
web/                         static single page app
config/                      regions.yaml, satellites.yaml, pipeline.yaml
data/                        eruptions.csv, lava volumes csv
.github/workflows/           live.yml (hourly), backfill.yml (dispatch), train.yml (dispatch)
docs/                        this spec, runbooks
tests/
```

Runtime dependencies: `eumdac`, `numpy`, `xarray`, `netCDF4`, `pyproj`,
`boto3`, `shapely` (point in polygon), `Pillow`, `pyyaml`. No satpy, dask or
torch at runtime.

### Live workflow (`live.yml`)

- Cron hourly. Also `workflow_dispatch`.
- Reads `state/<sat>.json` from the bucket (last processed timestamp).
- Searches the Data Store for products since then (bounded, e.g. max 24 h).
- For each new timestamp: Data Tailor cutout (image and cloud mask) →
  store raw → features → predict → anomaly products → store → append to the
  time series → update state. Each timestamp is idempotent: existing outputs
  are skipped unless `--force`.
- Finishes by rewriting `index.json` for the web app.
- Failures on a single timestamp are logged and skipped; state only advances
  past a timestamp once it has succeeded or has failed `max_attempts` times.

Note: scheduled Actions can be delayed and are disabled after 60 days without
repository activity on public repos; the catch-up logic handles both.

### Backfill workflow (`backfill.yml`)

`workflow_dispatch` with inputs: satellite key, start, end, cadence (e.g.
`1h`), mode (`raw-only` or `full`). Writes resumable progress to the bucket
so repeated dispatches continue where the previous run stopped. Used for the
training sample (cadence `sample`, driven by the stratified sampler) and for
the eruption validation periods (hourly). Both run within the 6 h job limit
and are re-dispatched as needed.

### Training workflow (`train.yml`)

`workflow_dispatch`; downloads the sampled raw cutouts, trains both
candidates, writes the evaluation report to `reports/` in the bucket and the
`.npz` artefacts to `models/`. Training may also be run locally.

## 9. Bucket layout and data licensing

The EUMETSAT terms of use prohibit redistribution of the original numerical
data. The project therefore uses **two Backblaze B2 buckets**:

- **Private bucket**: everything that is, or could be used to reconstruct,
  EUMETSAT data, plus internal pipeline state. Read and written only by the
  Action (and by the maintainer for training and analysis). Raw cutouts are
  never committed to the repository and never printed to Action logs.
- **Public bucket**: exactly the set of derived files the web app reads,
  nothing else. Public read, CORS enabled for the GitHub Pages origin.

Observed-channel quicklook PNGs in the public bucket are colour-mapped,
8-bit, fixed-scale renderings with a visible attribution line, not
recoverable as numerical data. **[verify]** against the exact licence text
that such quicklooks and the derived statistics are permitted; if in doubt,
drop the observed-channel quicklooks from the public bucket and show only
derived images.

### Private bucket

```
raw/<sat>/<YYYY>/<MM>/<DD>/<YYYYMMDDTHHMMSS>_l15.nc      Data Tailor image cutout
raw/<sat>/<YYYY>/<MM>/<DD>/<YYYYMMDDTHHMMSS>_clm.nc      cloud mask cutout
products/<sat>/<YYYY>/<MM>/<DD>/<YYYYMMDDTHHMMSS>/residual.npz   residual, zscore, flags
models/<sat>/<model_id>.npz, models/<sat>/latest.json
reports/<sat>/<model_id>/...                               evaluation outputs
timeseries/<sat>/<region>.parquet                         compacted series for analysis
state/<sat>.json                                           live pipeline state
state/backfill/<job_id>.json                               backfill progress
```

### Public bucket

```
products/<sat>/<YYYY>/<MM>/<DD>/<YYYYMMDDTHHMMSS>/{ir39,ir39_pred,residual,ir108,vis06,clm}.png
timeseries/<sat>/<region>/<YYYY>-<MM>.jsonl               one JSON object per line
index.json                                                 satellites, regions, time ranges, model ids
```

Because B2 objects are immutable, the "append" to a time series shard is a
read-modify-write of the per-month file. **[decide]** shard granularity once
volumes are known.

### Secrets and variables

Each bucket has its own application key restricted to that bucket, so a
leaked public-bucket key cannot read raw data.

Repository secrets:

- `EUMETSAT_CONSUMER_KEY`, `EUMETSAT_CONSUMER_SECRET`
- `B2_PRIVATE_KEY_ID`, `B2_PRIVATE_APPLICATION_KEY`
- `B2_PUBLIC_KEY_ID`, `B2_PUBLIC_APPLICATION_KEY`

Repository variables:

- `B2_PRIVATE_BUCKET`, `B2_PUBLIC_BUCKET`
- `B2_ENDPOINT` (S3 regional endpoint hostname, shared by both buckets)
- `B2_PUBLIC_URL` (base URL the web app reads from)

## 10. Web app

Static `web/index.html` plus one JS and one CSS file, hosted from GitHub
Pages via a `pages.yml` workflow. No build step.

- Loads `index.json`, then the time series shards for the selected
  satellite, region and date range.
- Time series plot (uPlot; Plotly is the fallback if interaction needs grow)
  with selectable statistic (`sum_pos`, `max`, `n_above`, `vrp_w`), night
  shading from `sza_mean`, markers for saturated samples, cloud fraction as a
  faint secondary trace, and eruption windows from `eruptions.csv` as shaded
  bands.
- Clicking a point loads the PNGs for that timestamp into an image panel
  (observed, predicted, residual, IR10.8, VIS0.6, cloud mask) with
  previous/next buttons stepping through time.
- URL state (satellite, region, range, selected time) so views can be shared.

## 11. Analysis: effusion rate model

Offline, in `analysis/`, Stan via `cmdstanpy`. Inputs: the anomaly time
series for each eruption window, lava volume measurements from
photogrammetric surveys (`data/lava_volumes.csv`: eruption, survey time,
cumulative volume, uncertainty, source), and the per-sample covariates
(`n_cloud`, `n_sat`, `sza_mean`).

Model sketch, fitted jointly across eruptions:

- Effusion rate `q(t) >= 0` on a regular grid (hourly) per eruption:
  `log q(t) = m(t) + f(t)`, where `m(t)` is a parametric mean (exponential
  decay from onset to a plateau, capturing the characteristic Reykjanes
  shape) and `f(t)` is a Gaussian process approximated with Hilbert space
  basis functions (fast, analytic in Stan).
- Cumulative volume `V(t)` by trapezoidal integration of `q` on the grid;
  survey volumes enter as Gaussian observations of `V(t_survey)`.
- Latent anomaly `a(t) = c * (k ⊛ q)(t)`: effusion rate convolved with a
  non-negative cooling kernel `k` defined on log-spaced lag bins spanning
  15 min to ~30 days (compare with a two-exponential kernel). `c` and `k`
  are shared across eruptions in the first iteration; hierarchical (partially
  pooled) per-eruption parameters with hyperpriors are the second iteration
  if pooling fits poorly.
- Observation model: observed statistic `y(t) <= a(t)` in expectation.
  One-sided noise anchored so that the best observations equal the latent
  value (cloud and terrain only reduce the signal), with the noise scale a
  function of `n_cloud` and `sza_mean`. Saturated samples are upper-censored
  (Tobit likelihood).

Outputs: posterior effusion rate curves per eruption, the shared
radiance-to-effusion coefficient, and the cooling kernel.

## 12. Milestones

| # | Milestone | Done when |
|---|---|---|
| M0 | Skeleton | package, config, CLI stubs, tests, CI lint; B2 and Data Store credentials smoke-tested from an Action |
| M1 | One cutout | `thermal fetch` retrieves one `msg11-0deg` image and cloud mask via Data Tailor in native projection, geolocates it, and stores it in `raw/` |
| M2 | Calendar and sampler | `eruptions.csv` curated; stratified sampler; backfill workflow collects the training sample |
| M3 | Model | both candidates trained; evaluation report; `.npz` in `models/`; residual noise on held-out nights documented |
| M4 | Products and live run | anomaly products, region statistics, time series; hourly Action running end to end |
| M5 | Web app | time series and image browsing live on GitHub Pages |
| M6 | Eruption backfill | hourly products for all Reykjanes eruptions 2021 to 2024; onset detection compared with the calendar |
| M7 | RSS and FCI | second and third satellite keys through the same pipeline |
| M8 | Effusion model | Stan model fitted to the M6 series with lava volume data |

M1 is the first thing to build because it resolves most of the **[verify]**
items.

## 13. Open items to verify in M1

- Collection ids, platform names and slot history in the Data Store.
- Data Tailor: native projection cutouts, output netCDF structure (do we get
  lat/lon arrays or projection coordinates), quotas, latency, and whether
  cloud mask products can be cut the same way.
- Saturation handling and flagging in the level 1.5 cutout.
- Radiance units and the conversion needed for the MIR method constant.
- Whether solar/satellite azimuth should be computed from first principles or
  read from product metadata.
