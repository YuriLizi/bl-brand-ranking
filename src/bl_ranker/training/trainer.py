"""Production trainer.

DESIGN
------
`bl_models_train.py` is vendored under `bl_ranker/original/` **byte-identical** to the
researcher's version, and this class subclasses it. Only I/O concerns are overridden:

  * where the input data comes from      (Delta table instead of a loose CSV)
  * where artifacts go                   (an MLflow run instead of a local folder)

Every line of feature engineering, every hyper-parameter and both model fits are
inherited unchanged. `git diff` against the researcher's file proves it -- there is no
need to take our word for it.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pandas as pd

from bl_ranker.config import get_settings
from bl_ranker.data.ingest import read_delta
from bl_ranker.original.bl_models_train import BLPayoutModelsFit
from bl_ranker.tabpfn_local import build_regressor

log = logging.getLogger(__name__)


class ProductionTrainer(BLPayoutModelsFit):
    """BLPayoutModelsFit wired to Delta input and MLflow output."""

    def __init__(
        self,
        output_predictors_path: str,
        train_test: bool,
        delta_version: int | None = None,
        use_hosted_tabpfn: bool | None = None,
    ):
        self.delta_version = delta_version
        self.use_hosted_tabpfn = use_hosted_tabpfn
        self._snapshot_dir = Path(tempfile.mkdtemp(prefix="bl_snapshot_"))
        self._gender_cache: dict[str, tuple[str, float]] = {}
        # Filled by the capture hooks below; consumed by run.py to log charts.
        self.captured: dict[str, object] = {}

        # Materialise the Delta slice to a CSV snapshot the vendored reader can open.
        # This is deliberate: the *table* stays the system of record (and we record its
        # version in MLflow), while the researcher's reader stays untouched. The cost is
        # one serialisation on a weekly batch job, which is irrelevant at this scale.
        snapshot_name = "bl_snapshot.csv"
        self._materialise_snapshot(self._snapshot_dir / snapshot_name)

        super().__init__(
            input_path=str(self._snapshot_dir) + "/",
            input_file=snapshot_name,
            output_predictors_path=output_predictors_path,
            train_test=train_test,
        )

    # ------------------------------------------------------------------ input
    def _materialise_snapshot(self, destination: Path) -> None:
        df = read_delta(version=self.delta_version)
        log.info(
            "Materialised %d rows from Delta (version=%s) to %s",
            len(df),
            self.delta_version if self.delta_version is not None else "latest",
            destination,
        )
        df.to_csv(destination, index=False)
        self.snapshot_rows = len(df)

    # ------------------------------------------------------------------ TabPFN
    def tabpfn_regression_payout(self, x_train_payout, y_train_payout):
        """Identical to the parent, except the regressor is local instead of hosted.

        The parent method calls `set_access_token(os.environ["TABPFN_TOKEN"])`, which
        hard-fails without a hosted-API token and pins us to a network dependency. The
        context construction and the `fit` call below are otherwise line-for-line the
        researcher's.
        """
        self.logger.info("start tabular transformer")
        CONTEXT_SIZE = 1000  # most recent context -- unchanged from the original
        # "batch": evaluation scores the whole test set in one call, where the
        # uncached mode is far cheaper. Serving uses "serving" instead.
        model_tfm = build_regressor(workload="batch", use_hosted=self.use_hosted_tabpfn)
        tfm_context = {
            "x": x_train_payout.iloc[-CONTEXT_SIZE:],
            "y": y_train_payout["payout"].iloc[-CONTEXT_SIZE:],
            "columns": list(x_train_payout.columns),
        }
        model_tfm.fit(tfm_context["x"], tfm_context["y"])
        self.logger.info("finish tabular transformer")
        return model_tfm, tfm_context

    # ------------------------------------------------------------- capture hooks
    # Each of these calls the researcher's own method and keeps a reference to what it
    # produced, so the run can chart it. Delegate-and-capture: nothing is recomputed and
    # no behaviour changes -- without this the fitted model and the test predictions are
    # local variables inside `fit_()` and die with it.

    def catbosot_model_sold_to_client(self, x_train, y_train):
        model = super().catbosot_model_sold_to_client(x_train, y_train)
        self.captured["cb_model"] = model
        self.captured["feature_names"] = list(x_train.columns)

        # TRAIN-set predictions. The researcher's evaluation reports the held-out week
        # only, so nothing in the original output shows whether the model is overfitting
        # -- and 800 trees at depth 8 over ~60k rows is a configuration where that
        # question is worth asking. Scoring the training set is a few seconds of CatBoost
        # and makes the train-versus-test gap visible.
        try:
            self.captured["train_y_true"] = y_train["sold_to_client"].to_numpy()
            self.captured["train_y_pred"] = model.predict(x_train)
        except Exception:  # pragma: no cover - never fail a run over a diagnostic
            log.warning("Could not capture train-set predictions", exc_info=True)
        return model

    def accuracy_classification_model(self, CB_model, y_test_all_days, x_test_all_days, target_col):
        result = super().accuracy_classification_model(
            CB_model, y_test_all_days, x_test_all_days, target_col
        )
        try:
            features = x_test_all_days.drop("split_day", axis=1)
            self.captured["y_true"] = y_test_all_days[target_col].to_numpy()
            self.captured["y_pred"] = CB_model.predict(features)
            # The probability, not just the class. expected_payout is P(lead) x payout, so
            # how well-calibrated P is directly determines whether the ranking is right --
            # a systematically over-confident P reorders brands even when the classes are
            # correct. Nothing in the researcher's report measures that.
            self.captured["y_proba"] = CB_model.predict_proba(features)[:, 1]
            # Which brand each test row is about, so performance can be broken down by
            # brand -- the dimension the whole product is organised around.
            if "client_name" in features.columns:
                self.captured["brand"] = features["client_name"].to_numpy()
        except Exception:  # pragma: no cover - a chart must never fail a training run
            log.warning("Could not capture classification predictions for plotting")
        return result

    def accuracy_cont_payout_prediction(self, model_tfm, x_test_payout, y_test_payout):
        result = super().accuracy_cont_payout_prediction(model_tfm, x_test_payout, y_test_payout)
        # The parent writes `pred_payout` onto y_test_payout in place.
        try:
            self.captured["payout_actual"] = y_test_payout["payout"].to_numpy()
            self.captured["payout_pred"] = y_test_payout["pred_payout"].to_numpy()
        except Exception:  # pragma: no cover
            log.warning("Could not capture payout predictions for plotting")
        return result

    # ------------------------------------------------------------------ gender
    def detect_gender_with_confidence(self, fname):
        """Memoised delegate to the researcher's own implementation.

        Training applies this lookup row-by-row over every registered session. First names
        repeat heavily across ~60k rows, so the same NameDataset search is repeated
        thousands of times. Calling the parent and caching its result keeps the values
        identical by construction while removing the redundant work.
        """
        cached = self._gender_cache.get(fname)
        if cached is None:
            cached = super().detect_gender_with_confidence(fname)
            self._gender_cache[fname] = cached
        return cached

    # ------------------------------------------------------------------ helpers
    @property
    def log_file(self) -> Path | None:
        """The researcher-defined log file, logged to MLflow for accuracy assessment."""
        candidates = sorted(Path(self.output_predictors_path).glob("log_file_bl_train_*.log"))
        return candidates[-1] if candidates else None
