"""Latency vs ranking-fidelity for TabPFN `n_estimators`, measured on real data.

`n_estimators` is the only remaining latency lever that actually changes predictions, so
the decision needs numbers rather than intuition. This script quantifies exactly what is
lost by shrinking TabPFN's internal ensemble:

  * per-request latency for one user (all brands), median of several repeats;
  * Spearman correlation of predicted payouts against the n_estimators=8 baseline;
  * top-1 agreement -- how often the *first* brand shown to the user is unchanged.

Top-1 agreement is the metric that matters commercially. The briefing states that users
tend to click the first brand in the list, so a configuration that reorders positions 7
and 8 costs nothing, while one that changes position 1 costs revenue.

    python benchmarks/tabpfn_fidelity.py --users 25 --out benchmarks/fidelity.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scipy.stats import spearmanr  # noqa: E402
from tabpfn import TabPFNRegressor  # noqa: E402

from bl_ranker.training.trainer import ProductionTrainer  # noqa: E402

CONTEXT_SIZE = 1000  # the researcher's setting; not varied here


def build_real_data(n_users: int, seed: int):
    """Run the researcher's own preprocessing to get a genuine context and query set."""
    workdir = tempfile.mkdtemp(prefix="bl_fidelity_")
    trainer = ProductionTrainer(output_predictors_path=workdir + "/", train_test=False)
    x_train, y_train, _, _ = trainer.bl_preprocessing()
    x_payout, y_payout = trainer.prepare_for_cont_payout_prediction_train(x_train, y_train)

    context_x = x_payout.iloc[-CONTEXT_SIZE:]
    context_y = y_payout["payout"].iloc[-CONTEXT_SIZE:]

    brands = [b for b in x_train["client_name"].dropna().unique() if b != "other"]

    # One query block per user: that user's features replicated across every brand,
    # which is exactly the shape the predictor builds at request time.
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(x_payout), size=min(n_users, len(x_payout)), replace=False)
    blocks = []
    for idx in picks:
        row = x_payout.iloc[idx]
        block = pd.DataFrame([row] * len(brands)).reset_index(drop=True)
        block["client_name"] = brands
        blocks.append(block)

    return context_x, context_y, brands, blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--estimators", type=int, nargs="+", default=[8, 4, 2, 1])
    parser.add_argument("--out", type=Path, default=Path("benchmarks/fidelity.json"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    context_x, context_y, brands, blocks = build_real_data(args.users, args.seed)
    n_brands = len(brands)
    queries = pd.concat(blocks, ignore_index=True)

    print(f"context      : {context_x.shape[0]} rows x {context_x.shape[1]} features")
    print(f"payout range : ${context_y.min():.0f} - ${context_y.max():.0f}")
    print(f"brands ({n_brands:2d})  : {brands}")
    print(f"query set    : {queries.shape[0]} rows ({len(blocks)} users x {n_brands} brands)\n")

    results: dict[int, dict] = {}
    for n_est in args.estimators:
        model = TabPFNRegressor(
            ignore_pretraining_limits=True,
            device="cpu",
            n_estimators=n_est,
            fit_mode="fit_with_cache",
            random_state=42,
        )
        start = time.perf_counter()
        model.fit(context_x, context_y)
        fit_seconds = time.perf_counter() - start

        latencies = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            model.predict(blocks[0])
            latencies.append((time.perf_counter() - start) * 1000)

        results[n_est] = {
            "fit_seconds": round(fit_seconds, 1),
            "latency_ms_p50": round(float(np.median(latencies)), 1),
            "predictions": model.predict(queries),
        }
        print(
            f"n_estimators={n_est}: startup fit {fit_seconds:6.1f}s | "
            f"per-request p50 {np.median(latencies):8.1f} ms"
        )

    baseline_key = max(results)
    baseline = results[baseline_key]["predictions"]

    print(f"\n=== fidelity against n_estimators={baseline_key} ===")
    header = f"{'n_est':>6} {'lat_ms':>9} {'speedup':>9} {'spearman':>10} {'top1_same':>11} {'mean_abs_$':>11}"
    print(header)
    print("-" * len(header))

    table = []
    for n_est in args.estimators:
        preds = results[n_est]["predictions"]
        rho = float(spearmanr(baseline, preds).statistic)
        top1 = float(
            np.mean([
                np.argmax(baseline[i * n_brands:(i + 1) * n_brands])
                == np.argmax(preds[i * n_brands:(i + 1) * n_brands])
                for i in range(len(blocks))
            ])
        )
        mean_abs = float(np.mean(np.abs(baseline - preds)))
        speedup = results[baseline_key]["latency_ms_p50"] / results[n_est]["latency_ms_p50"]
        row = {
            "n_estimators": n_est,
            "fit_seconds": results[n_est]["fit_seconds"],
            "latency_ms_p50": results[n_est]["latency_ms_p50"],
            "speedup_vs_baseline": round(speedup, 1),
            "spearman_vs_baseline": round(rho, 4),
            "top1_agreement": round(top1, 4),
            "mean_abs_payout_diff": round(mean_abs, 3),
        }
        table.append(row)
        print(
            f"{n_est:>6} {row['latency_ms_p50']:>9.1f} {speedup:>8.1f}x "
            f"{rho:>10.4f} {top1:>10.0%} {mean_abs:>11.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "hardware": "CPU-only",
                "context_size": CONTEXT_SIZE,
                "n_brands": n_brands,
                "n_users": len(blocks),
                "baseline_n_estimators": baseline_key,
                "results": table,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
