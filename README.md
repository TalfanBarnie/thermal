# thermal

Lightweight detection of volcanic thermal anomalies over Iceland from
Meteosat (SEVIRI and FCI) imagery, designed to run in a GitHub Action with a
Backblaze B2 bucket for storage and a static web app for display.

The design is described in [docs/SPEC.md](docs/SPEC.md).

## Layout

- `thermal/` runtime package (numpy-only inference; no satpy or torch at runtime)
- `config/` satellites, regions and pipeline settings
- `docs/SPEC.md` design specification and milestones
- `.github/workflows/` `ci` (lint and tests), `smoke test` and `fetch one timestamp`
  (both run manually from the Actions tab)

## Running locally

```bash
pip install -e ".[dev]"
pytest
```

Credentials are read from the environment using the names in the spec:
`EUMETSAT_CONSUMER_KEY`, `EUMETSAT_CONSUMER_SECRET`, `B2_ENDPOINT`,
`B2_PRIVATE_BUCKET`, `B2_PRIVATE_KEY_ID`, `B2_PRIVATE_APPLICATION_KEY`,
`B2_PUBLIC_BUCKET`, `B2_PUBLIC_KEY_ID`, `B2_PUBLIC_APPLICATION_KEY`, `B2_PUBLIC_URL`.

```bash
thermal smoke                                   # check buckets and Data Store access
thermal fetch --sat msg-0deg --time 2024-03-16T22:00:00Z --out ./scratch
thermal inspect ./scratch/l15/*.nc              # metadata summary of a cutout
```

Raw EUMETSAT cutouts must never be committed; `*.nc` is ignored by git.

## Licence and data

The code in this repository is released under the [MIT licence](LICENSE),
copyright 2026 Talfan Barnie. The licence covers the code and configuration
only; the material below has its own terms.

- **Satellite data.** The pipeline reads Meteosat SEVIRI level 1.5 imagery and
  cloud masks from the EUMETSAT Data Store under EUMETSAT's data licence. The
  raw cutouts are kept in a private bucket and are never committed or
  published. The images and time series on the public site are derived
  products (model residuals, statistics and fixed-scale colour renderings)
  and are not a redistribution of the original data; they are published with
  the attribution "contains modified EUMETSAT data". Running the pipeline
  yourself requires your own EUMETSAT Data Store account.
- **Eruption calendar.** `data/eruptions.csv` was compiled from the
  Smithsonian Institution Global Volcanism Program (Volcanoes of the World
  database) and from Wikipedia eruption articles (text licensed CC BY-SA 4.0),
  with manual corrections. Please cite those sources if you reuse it.
- **uPlot.** The web app vendors [uPlot](https://github.com/leeoniya/uPlot),
  MIT licensed; its licence file is in `web/vendor/`.

