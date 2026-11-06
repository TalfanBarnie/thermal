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
thermal fetch --sat msg11-0deg --time 2024-03-16T22:00:00Z --out ./scratch
thermal inspect ./scratch/l15/*.nc              # metadata summary of a cutout
```

Raw EUMETSAT cutouts must never be committed; `*.nc` is ignored by git.
