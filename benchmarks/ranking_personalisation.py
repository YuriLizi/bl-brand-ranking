"""Does the ranking actually personalise, or is it a static brand order?

The feature-importance chart showed `client_name` carrying ~40% of total importance,
6.6x the next feature. That is expected -- brands have very different base acceptance
rates -- but it raises the question the product depends on: does the model use the SURVEY
ANSWERS to reorder brands per user, or does every user get roughly the same list?

This measures it, and changes nothing. Two experiments:

  1. SHAP importance, which attributes per-prediction contribution rather than CatBoost's
     default split-based metric (that metric is known to inflate high-cardinality
     categoricals, and `client_name` is exactly that).
  2. Rank variation across deliberately extreme user profiles. If a 720+/$200k business
     and a sub-550/$5k business get the same top brand, the ranking is static in the way
     that matters -- users click the first item.

Run:  python benchmarks/ranking_personalisation.py
"""
from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import mlflow  # noqa: E402
from catboost import CatBoostClassifier  # noqa: E402

from bl_ranker.config import get_settings  # noqa: E402
from bl_ranker.example_user import EXAMPLE_USER  # noqa: E402

# Deliberately opposite ends of every survey axis the briefing names.
PROFILES = {
    "strong": {
        "credit_score": "Excellent (720+)",
        "monthly_revenue": "$200,000+",
        "loan_amount": "$100,000 - $199,999",
        "time_in_business": "2+ years",
        "business_type": "S Corporation",
        "industry": "construction",
    },
    "weak": {
        "credit_score": "Very Poor - Under 550",
        "monthly_revenue": "$0 - $9,999",
        "loan_amount": "$10,000 - $24,999",
        "time_in_business": "Less than 6 months",
        "business_type": "Other",
        "industry": "retail_trade",
    },
    "mid": {
        "credit_score": "Fair (600-649)",
        "monthly_revenue": "$20,000 - $49,999",
        "loan_amount": "$25,000 - $49,999",
        "time_in_business": "1-2 years",
        "business_type": "LLC",
        "industry": "other",
    },
}


def load_champion_artifacts() -> tuple[CatBoostClassifier, pd.DataFrame]:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    mv = client.get_model_version_by_alias(
        settings.registered_model_name, settings.champion_alias
    )
    local = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=mv.source, dst_path=tempfile.mkdtemp()
        )
    )
    cbm = next(local.rglob("*.cbm"))
    clients = next(local.rglob("all_clients.csv"))
    model = CatBoostClassifier(allow_writing_files=False).load_model(str(cbm), format="cbm")
    return model, pd.read_csv(clients)


def build_features(profile: dict, brands: list[str]) -> pd.DataFrame:
    """Run one profile through the researcher's own preprocessing, across every brand."""
    from bl_ranker.serving.predictor import WarmPredictor

    user = dict(EXAMPLE_USER)
    user.update(profile)

    predictor = WarmPredictor.__new__(WarmPredictor)  # no model load needed
    predictor._gender_cache = {}
    predictor.user_data = user
    predictor._all_clients = pd.DataFrame({"client_name": brands})
    predictor.logger = __import__("logging").getLogger("diag")
    return predictor.bl_preprocessing()


def main() -> None:
    model, all_clients = load_champion_artifacts()
    brands = [b for b in all_clients["client_name"].dropna().unique() if b != "other"]
    print(f"champion CatBoost loaded | {len(brands)} brands\n")

    # ---------------------------------------------------------------- SHAP
    print("=" * 72)
    print("SHAP importance (per-prediction attribution)")
    print("=" * 72)
    frames = [build_features(p, brands) for p in PROFILES.values()]
    sample = pd.concat(frames, ignore_index=True)
    from catboost import Pool

    cat_cols = sample.select_dtypes(include=["object", "category"]).columns.tolist()
    pool = Pool(sample, cat_features=cat_cols)
    shap = model.get_feature_importance(pool, type="ShapValues")
    contrib = np.abs(shap[:, :-1]).mean(axis=0)
    order = np.argsort(contrib)[::-1]
    total = contrib.sum()
    for i in order[:12]:
        print(f"  {sample.columns[i]:26} {contrib[i]:8.4f}  ({contrib[i]/total:5.1%})")

    # ------------------------------------------------------- rank variation
    print()
    print("=" * 72)
    print("Does the brand order change between user profiles?")
    print("=" * 72)
    rankings: dict[str, list[str]] = {}
    for name, profile in PROFILES.items():
        feats = build_features(profile, brands)
        proba = model.predict_proba(feats)[:, 1]
        ranked = (
            pd.DataFrame({"brand": feats["client_name"], "p_lead": proba})
            .sort_values("p_lead", ascending=False)
            .reset_index(drop=True)
        )
        rankings[name] = ranked["brand"].tolist()
        print(f"\n  {name.upper():8} top 5 by P(lead):")
        for _, row in ranked.head(5).iterrows():
            print(f"     {row['brand']:26} {row['p_lead']:.4f}")

    print()
    print("  pairwise comparison:")
    for a, b in itertools.combinations(rankings, 2):
        ra, rb = rankings[a], rankings[b]
        same_top = ra[0] == rb[0]
        from scipy.stats import spearmanr

        pos_a = {brand: i for i, brand in enumerate(ra)}
        rho = spearmanr([pos_a[x] for x in rb], range(len(rb))).statistic
        print(f"     {a:7} vs {b:7}  top-1 same: {str(same_top):5}  rank corr: {rho:+.3f}")

    identical = len({tuple(v) for v in rankings.values()}) == 1
    print()
    print(f"  ALL THREE RANKINGS IDENTICAL: {identical}")
    out = Path("benchmarks/personalisation.json")
    out.write_text(json.dumps({
        "rankings": rankings,
        "identical": identical,
        "shap_top": [
            {"feature": sample.columns[i], "mean_abs_shap": float(contrib[i]),
             "share": float(contrib[i] / total)} for i in order[:12]
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
