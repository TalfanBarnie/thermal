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
- Data *sources* are keyed by slot/service (`msg-0deg`, `msg-rss`,
  `mtg-0deg`) in `config/satellites.yaml`. The platform occupying a slot
  changes over the archive (M1 found Meteosat-10/MSG3 serving 0° in March
  2024), so the platform is read from each product's metadata and stored
  with the data rather than fixed in config. Background models are trained
  per (source, platform) pair, e.g. `msg-0deg/MSG3`.
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
notes`. Compiled (M2) from the Global Volcanism Program eruption web service
and Wikipedia wikitext fetched by `scripts/scrape_eruptions.py` in the
`scrape eruption tables` workflow, then hand-curated and committed. Each
Sundhnúkur fissure eruption is its own row; a few end dates are marked
"verify" in the notes, which the 30 day buffer makes immaterial for training. Must cover at least Eyjafjallajökull 2010, Grímsvötn 2011,
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

## 13. Milestone M1 findings (2026-11-06)

First cutout: `msg-0deg`, nominal time 2024-03-16T22:00Z (night, Sundhnúkur
eruption in progress). Resolved items:

- **Platform**: the 0° collection served Meteosat-10 (`MSG3`) in March 2024,
  not Meteosat-11. Sources are therefore keyed by slot and the platform is
  recorded from product metadata (section 3).
- **Data Tailor region**: `RegionOfInterest(NSWE=[N, S, W, E])` must be a
  list of four numbers; a comma string is rejected.
- **Projection**: with no `projection` in the chain the output stays on the
  native geostationary grid, with a CF `geostationary` grid mapping variable
  and `x`/`y` projection coordinates in metres (GRS80 ellipsoid, height
  35 785 831 m, sweep `y`). Pixel spacing is ~2993 m by ~2997 m in projection
  units; the ground footprint over Iceland is ~42 km² median.
- **Cutout size**: 45 rows by 199 columns, 0.43 MB for the image and 0.06 MB
  for the cloud mask. Pixels outside the lat/lon box but inside its
  projection-space bounding box are NaN (about 38%).
- **Grid identity**: the cloud mask cutout has exactly the same
  `GeoTransform` and shape as the image cutout, so no resampling is needed.
- **Variables**: image channels are `channel_1` to `channel_11` in SEVIRI
  band order (VIS0.6, VIS0.8, NIR1.6, IR3.9, WV6.2, WV7.3, IR8.7, IR9.7,
  IR10.8, IR12.0, IR13.4); HRV is excluded by the Data Tailor. Values are
  calibrated radiances (offset/slope in the `chNN_cal` global attributes).
  The cloud mask variable is `cloud_mask` with values 0 to 2 (MSG CLM
  convention: 0 clear over water, 1 clear over land, 2 cloud; 3/no data
  appears as NaN **[verify]**).
- **Timing**: 106 s for the image customisation and 45 s for the cloud
  mask, including queueing and download; ~2.5 min per timestamp sequentially.
  Data Tailor workspace quota is 20 GB; customisations are deleted after
  download.
- **Idempotency**: `thermal fetch` now reuses a cutout already in the
  private bucket instead of re-running the Data Tailor.
- **Signal check**: the IR3.9 quicklook shows one isolated bright pixel
  cluster at the Reykjanes peninsula, absent from IR10.8, with the rest of
  the scene cloud-covered. The eruption is detectable through cloud in a
  single raw channel, before any modelling.
- **Cloud mask (resolved in M2)**: a daytime scene (2023-07-12T13:00Z)
  shows all three classes (0 clear sea, 1 clear land, 2 cloud) and the
  footprint check (`inspect.footprint_check`) confirms the finite pixels of
  both cutouts span the same lat/lon box, so the grids agree. The cloud mask
  simply has no data in a band along the north-west edge of the box (49% vs
  62% finite), consistent with a satellite zenith angle limit in the cloud
  mask product. Class 3 / no data is encoded as NaN.

## 14. Milestone M2 findings (2026-11-06)

- **Platform history** of the 0° slot (from EUMETSAT, OSI SAF and NOAA
  notices; `data/platform_history.csv`): Meteosat-8 2004 to 2007, Meteosat-9
  2007 to 21 Jan 2013, Meteosat-10 to 20 Feb 2018, Meteosat-11 to 21 Mar
  2023, Meteosat-10 since. Fagradalsfjall 2021 and 2022 were imaged by
  Meteosat-11 (`msg-0deg/MSG4`); Litli-Hrútur 2023 and every Sundhnúkur
  eruption by Meteosat-10 (`msg-0deg/MSG3`). Two models cover the validation
  eruptions.
- **Training samples** (`data/samples/`): 1000 timestamps each for the
  MSG3 era (2023-03-22 onwards) and the MSG4 era (2018-02-21 to
  2023-03-20), stratified over month and hour, eruption windows excluded
  with the 30 day buffer. The 2013 to 2018 Meteosat-10 era is a reserve
  source of extra MSG3 training data if needed.
- **Data Tailor throughput**: ~2.5 min per timestamp (image plus cloud
  mask) sequentially; the backfill workflow runs 2 customisations at a time
  in 330 minute slices and resumes from state in the private bucket.
- **Eruption calendar**: `data/eruptions.csv`, 18 rows, from the GVP web
  service and Wikipedia wikitext via `scripts/scrape_eruptions.py`.

## 15. Milestone M3 status (2026-11-06)

Code complete and exercised end to end on the first 29 cutouts of the MSG3
training sample (all from March and April 2023, since the first backfill
slice ran in date order; later slices run in a fixed shuffled order).

- `thermal/features.py` builds the 26-feature matrix (10 radiances, 7
  brightness temperatures, solar zenith and relative azimuth, viewing
  zenith, lat, lon, day-of-year) identically for training and inference.
- `training/` fits a degree-2 polynomial ridge and a 3x128 GELU MLP, both
  predicting mean and log-variance, evaluates on held-out whole days, and
  exports numpy `.npz` artefacts that `thermal/model.py` runs with no torch.
- `thermal predict` applies the bucket's latest model to a timestamp and
  writes residual, z-score, observed and predicted quicklooks plus
  monitoring-region statistics.
- First numbers (29 images, 20% of days held out; a smoke test, not a
  result): MLP test RMSE 3.7 K, ridge 4.3 K, in IR3.9 brightness
  temperature; night-time training residual 1.6 K RMSE. The z-score spread
  on held-out days was 1.4, so the variance head is over-confident at this
  data volume.
- Detection check on 2024-03-16T22:00Z (Meteosat-10, fully cloudy): the
  residual image is flat apart from the eruption, maximum z-score 13.9,
  8 pixels above z=4 in the Reykjanes region. Cloud structure that
  dominated the raw IR3.9 image is removed by the model.
- Remaining for M3: retrain once the backfills complete (~1000 images per
  era), compare the candidates per solar-zenith bin, check the residual
  noise floor at night against the instrument NEdT, and add the spatial
  mean-residual map and histogram figures to the report.

## 16. Milestone M4 status (2026-11-06)

- `thermal/products.py`: per timestamp, `residual.npz` (residual, z-score,
  predicted std, flags, lat, lon) in the private bucket and fixed-scale PNGs
  (ir39, ir39_pred, residual, zscore, ir108, vis06 by day only, clm) in the
  public bucket under `products/<sat>/<Y>/<M>/<D>/<stamp>/`. Flags: 1 invalid,
  2 saturated IR3.9 (radiance at count 1023 from the `chNN_cal` attribute),
  4 cloud, 8 cloud mask unavailable.
- `thermal/timeseries.py`: one row per region and timestamp with `sum_pos`,
  `max`, `max_z`, `n_above` (z > 4), `mean`, `vrp_w` (MIR method with the
  per-wavenumber to per-micron conversion, **[verify]** the constant and
  units), `n_sat`, `n_cloud`, `n_cloud_unknown`, `sza_mean`, platform and
  model id; monthly `.jsonl` shards in the public bucket updated by
  read-modify-write; `index.json` listing sources, regions and months.
- `thermal/live.py`: `process_timestamp` (fetch or reuse, model, products,
  rows) and `run_live` (candidates since the stored state, bounded by a
  lookback and a maximum count; nominal times are parsed from product ids so
  no per-product requests are made; state and index updated after each).
- Workflows: `live` (hourly cron, concurrency group, manual dispatch) and
  `process one timestamp`.
- Models are loaded per platform from `models/<sat>/<platform>/latest.json`;
  a timestamp whose platform has no model is recorded as an error and left
  for a later run.
- **Near-real-time embargo (found in M4)**: the Data Store returns 403 for
  image products younger than about one hour for this account (the cloud
  mask has no such limit). `thermal probe` measures it; the live loop only
  considers nominal times older than `live.min_age_minutes` (75) in
  `config/pipeline.yaml`, and a 403 is not retried.
- **Verified end to end** on 2026-11-06: three evening timestamps fetched,
  modelled, published to `products/` and `timeseries/msg-0deg/reykjanes/2026-11.jsonl`,
  state and `index.json` updated. The rows carry nonsense values (mean
  residual −0.69, z-scores in the thousands) because the 29-image smoke
  model extrapolates badly to a November night; they will be replaced once
  the full model exists. Rows are keyed by time, so reprocessing overwrites.
- **Scheduled runs need the default branch**: GitHub only fires cron
  schedules from the repository's default branch, so the hourly `live`
  workflow starts once this branch is merged (or made the default). Until
  then it can be dispatched manually.
- **Still to do**: a `reprocess` mode that re-runs existing products with a
  newer model (needed after every retrain and for the eruption backfill),
  and the parquet compaction of the shards for analysis.

## 17. Milestone M5 status (2026-11-06)

- `web/`: `index.html`, `app.js`, `style.css`, `config.js` and a vendored
  uPlot 1.6.30 (MIT) so the page has no third-party runtime dependency. No
  build step.
- Reads `index.json`, the monthly shards of the chosen source and region, and
  `data/eruptions.csv` (copied into the site at deploy). Filters in one row:
  range presets, source, region, statistic. One axis per chart: the chosen
  statistic above, cloud fraction below, cursors synced. Night shading from
  the mean solar zenith, eruption bands from the calendar, saturated samples
  marked, crosshair tooltip with every value, a table view, lines and bands
  broken at data gaps longer than 3 hours, URL hash state, selected light
  and dark themes with a toggle, phone layout without horizontal scroll.
- Clicking a point (or a table row, or arrow keys) shows the product PNGs
  for that time with previous/next navigation; the visible image is omitted
  at night.
- `pages` workflow publishes `web/` to GitHub Pages, writing `config.js`
  from the `B2_PUBLIC_URL` variable. It needs Pages enabled with source
  "GitHub Actions" in the repository settings and runs on pushes to the
  default branch or on dispatch.
- Verified locally in headless Chromium against the live bucket at desktop
  and phone widths in both themes: no console errors, no overflow.

## 18. Milestone M6 status (2026-11-06, in progress)

- `thermal backfill --mode process [--force]`: the resumable batch runner
  now also models and publishes (products, rows, index). With `--force` it
  re-makes existing products, which is the reprocess path after a retrain.
- `thermal eruption-windows`: timestamps covering every calendar eruption
  since 2021 with a 2 day margin, 3-hourly, plus hourly for the first 3 days
  of each eruption: 3976 timestamps in
  `data/samples/msg-0deg_eruptions.csv`. Fetching them is the `eruptions`
  backfill job (fetch mode); processing follows once the full models exist.
- `thermal validate`: reads the public shards over HTTP, reports per
  eruption the first sample above a threshold and its delay from the
  calendar start, false alarms outside the buffered windows, and exports one
  CSV per eruption window for the effusion-rate analysis.
- Pending: retrain both eras on the full samples, choose the candidates,
  process the eruption windows and reprocess the live rows, then run the
  validation and record the results here.

## 19. Remaining items to verify

- Exact handover times in `data/platform_history.csv` (the backfill records
  the platform of every fetched product, which will confirm them).
- RSS and FCI collection ids and Data Tailor product types.
- Saturation handling and flagging in the level 1.5 cutout.
- Cloud mask value semantics, in particular how no-data is encoded.
- Radiance units and the conversion needed for the MIR method constant.
- Whether solar/satellite azimuth should be computed from first principles or
  read from product metadata.
