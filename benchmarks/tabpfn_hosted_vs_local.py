"""Hosted TabPFN API vs local open-source weights: latency and fidelity.

WHY THIS MATTERS
----------------
The researcher's scripts call TabPFN through `tabpfn-client`, a client for Prior Labs'
HOSTED api -- so the hosted path is the assignment's baseline, not an alternative. We ship
local open-source weights by default and keep the hosted path behind
`USE_HOSTED_TABPFN=true`. This script measures what that swap actually costs and buys,
because the briefing asks to productize TabPFN "in the most efficient way" and that claim
needs a number on both sides.

Two questions, both answered on the SAME context and the SAME query rows:

  1. LATENCY -- hosted does a network round-trip per predict; local does a CPU forward
     pass. Hosted runs on the provider's GPU, so it is genuinely unclear a priori which
     wins on a 10-row request. That is the point of measuring.
  2. FIDELITY -- hosted is TabPFN v3.0, the open-source weights are v2. Predictions are
     close but not identical. What matters commercially is not the payout error but
     whether the TOP-RANKED BRAND changes, because the briefing says users click first.

WHAT IS TRANSMITTED
-------------------
The hosted path uploads the 1000-row context. Those 25 columns are engineered features,
not raw session data: no names (only `fname_len`/`lname_len`), no phone number (only
`cellphone_prefix`), no address. Still third-party data egress, and worth stating.

Deliberately modest request counts -- this spends the user's API quota.
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> None:
    """Minimal .env reader -- avoids a python-dotenv dependency for one variable."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def timed_predict(model, frame: pd.DataFrame, repeats: int) -> tuple[list[float], np.ndarray]:
    latencies: list[float] = []
    last = None
    for _ in range(repeats):
        started = time.perf_counter()
        last = model.predict(frame)
        latencies.append((time.perf_counter() - started) * 1000)
    return latencies, np.asarray(last, dtype=float)


def summarise(name: str, latencies: list[float]) -> None:
    latencies = sorted(latencies)
    p = lambda q: latencies[min(len(latencies) - 1, int(q / 100 * len(latencies)))]
    print(
        f"  {name:<22} n={len(latencies):<3} "
        f"p50={statistics.median(latencies):>8.1f}ms  "
        f"p95={p(95):>8.1f}ms  min={latencies[0]:>8.1f}ms  max={latencies[-1]:>8.1f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="payout_tfm_context.joblib")
    parser.add_argument("--rows", type=int, default=10, help="query rows per call (brands)")
    parser.add_argument("--repeats", type=int, default=15, help="predict calls per backend")
    parser.add_argument("--skip-hosted", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    args = parser.parse_args()

    load_env(REPO_ROOT / ".env")

    ctx = joblib.load(args.context)
    x, y, columns = ctx["x"], ctx["y"], ctx["columns"]
    x = pd.DataFrame(x, columns=columns) if not isinstance(x, pd.DataFrame) else x
    query = x.iloc[: args.rows].copy()

    print(f"context: {x.shape[0]} rows x {x.shape[1]} features")
    print(f"query:   {args.rows} rows per call, {args.repeats} calls per backend\n")

    results: dict[str, np.ndarray] = {}

    if not args.skip_local:
        from bl_ranker.tabpfn_local import build_regressor

        print("LOCAL (open-source v2 weights, CPU, fit_with_cache)")
        model = build_regressor(workload="serving", use_hosted=False)
        t0 = time.perf_counter()
        model.fit(x, y)
        print(f"  fit (once, at startup): {time.perf_counter() - t0:.1f}s")
        lat, preds = timed_predict(model, query, args.repeats)
        summarise("local predict", lat)
        results["local"] = preds
        print()

    if not args.skip_hosted:
        if not os.environ.get("TABPFN_TOKEN"):
            print("HOSTED: skipped -- TABPFN_TOKEN not set")
        else:
            from bl_ranker.tabpfn_local import build_regressor

            print("HOSTED (Prior Labs API, v3.0, their GPU)")
            model = build_regressor(use_hosted=True)
            t0 = time.perf_counter()
            model.fit(x, y)
            print(f"  fit (context upload):   {time.perf_counter() - t0:.1f}s")
            lat, preds = timed_predict(model, query, args.repeats)
            summarise("hosted predict", lat)
            results["hosted"] = preds
            print()

    if len(results) == 2:
        a, b = results["local"], results["hosted"]
        order_local = np.argsort(-a)
        order_hosted = np.argsort(-b)
        print("FIDELITY (v2 local vs v3 hosted, same context and query)")
        print(f"  mean abs diff:        ${np.abs(a - b).mean():.4f}")
        print(f"  max  abs diff:        ${np.abs(a - b).max():.4f}")
        print(f"  mean predicted payout: local ${a.mean():.2f}  hosted ${b.mean():.2f}")
        print(f"  top-1 row identical:  {order_local[0] == order_hosted[0]}")
        print(f"  ranking identical:    {np.array_equal(order_local, order_hosted)}")
        if len(a) > 2:
            from scipy.stats import spearmanr

            print(f"  Spearman correlation: {spearmanr(a, b).statistic:.4f}")


if __name__ == "__main__":
    main()
