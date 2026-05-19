# Profiling baselines

`baseline.json` is a committed timing snapshot captured by `scripts/profiling/profile_export.py`.
It records `export_frame()` latency stats (median / p95 / p99 in µs) at the reference
configuration (1920×1080 rgba8, 2 slots, 1000 frames after 50-frame warmup).

## Generating or refreshing the baseline

Requires a CUDA GPU and cuda-link installed (or `src/` on `sys.path`):

```powershell
# From repo root:
python scripts/profiling/profile_export.py --frames 1000 --outfile .profiling/baseline.json
```

After running, commit the updated `baseline.json` so CI has a reference point.

## Comparing a PR against the baseline

```powershell
python scripts/profiling/profile_export.py --frames 1000 --outfile .profiling/current.json
python scripts/profiling/compare.py .profiling/baseline.json .profiling/current.json
```

Exit code 1 = regression beyond 10% threshold on any region.

## Full scalene line-level profile

```powershell
# Install once: pip install scalene
python -m scalene --cli --json --outfile .profiling/scalene_current.json -- `
    scripts/profiling/profile_export.py --frames 1000
```

Scalene output is **not** committed — `scalene_*.json` is in `.gitignore`.
Use it for deep dives; `baseline.json` is the CI-stable artifact.
