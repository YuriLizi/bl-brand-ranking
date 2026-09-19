"""The composite MLflow pyfunc model.

WHY ONE MODEL INSTEAD OF THREE ARTIFACTS
----------------------------------------
Serving a ranking needs three things that must agree with each other: the CatBoost
classifier, the TabPFN context (whose feature columns must match what CatBoost was
trained on), and the brand universe. Registering them separately would make the question
"which combination was serving last Tuesday?" unanswerable, and a rollback would mean
hand-picking three compatible versions.

Packaging them as ONE pyfunc means one registry version == one atomic, self-consistent
serving state. Rollback is `set_registered_model_alias(champion, <older version>)` and
nothing else.

The expensive work -- fitting TabPFN's in-context training set -- happens in
`load_context`, which MLflow calls once when the model is loaded, not per prediction.
"""
from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import mlflow.pyfunc
import pandas as pd
from catboost import CatBoostClassifier

log = logging.getLogger(__name__)

ARTIFACT_CATBOOST = "catboost_model"
ARTIFACT_TFM_CONTEXT = "tfm_context"
ARTIFACT_ALL_CLIENTS = "all_clients"


class BrandRankerModel(mlflow.pyfunc.PythonModel):
    """Ranks the brand universe for one user by expected payout."""

    def __getstate__(self):
        """Never serialise runtime state.

        MLflow loads a freshly logged model and runs the `input_example` through it to
        infer a signature. That calls `load_context`, which populates `_predictor` with
        the fitted TabPFN -- torch weights and all -- and the instance is then pickled
        *with* it. The result was a **1 GB** `python_model.pkl` that OOM-killed the MLflow
        artifact server the moment a serving container tried to download it:

            Worker (pid:16) was sent SIGKILL! Perhaps out of memory?

        `_predictor` is rebuilt by `load_context` on every load, so excluding it loses
        nothing and keeps the registered model a few kilobytes.
        """
        state = self.__dict__.copy()
        state.pop("_predictor", None)
        return state

    @staticmethod
    def _artifact(context, key: str) -> str:
        """Resolve an artifact path, tolerating separators from the other OS.

        MLflow writes the artifact sub-path into MLmodel using the separator of whichever
        machine logged the model. A model registered on Windows records
        `artifacts\\payout_tfm_context.joblib`; on Linux that backslash is not a separator
        but part of the filename, so loading fails with:

            FileNotFoundError: '/tmp/tmp89nd5l37/artifacts\\payout_tfm_context.joblib'

        That makes a Windows-registered model unloadable in a Linux container -- which is
        precisely the portability the container is meant to provide. Rather than forbid
        registering from Windows, we repair the path at load time: try it as given, then
        fall back to the basename inside the artifacts directory.
        """
        raw = context.artifacts[key]
        if Path(raw).exists():
            return raw

        repaired = raw.replace("\\", "/")
        if Path(repaired).exists():
            log.warning("Repaired Windows-style artifact path for %r", key)
            return repaired

        # Last resort: find the file by name anywhere under the model directory.
        name = PurePosixPath(repaired).name
        root = Path(repaired).parent
        while root != root.parent and not root.exists():
            root = root.parent
        for candidate in root.rglob(name):
            log.warning("Located artifact %r by name at %s", key, candidate)
            return str(candidate)

        raise FileNotFoundError(f"Artifact {key!r} not found (recorded as {raw!r})")

    def load_context(self, context):
        from bl_ranker.config import get_settings
        from bl_ranker.serving.predictor import WarmPredictor
        from bl_ranker.tabpfn_local import build_regressor, configure_torch_threads

        settings = get_settings()
        configure_torch_threads(settings.torch_num_threads)

        tfm_context = joblib.load(self._artifact(context, ARTIFACT_TFM_CONTEXT))
        all_clients = pd.read_csv(self._artifact(context, ARTIFACT_ALL_CLIENTS))

        cb_model = CatBoostClassifier(allow_writing_files=False).load_model(
            self._artifact(context, ARTIFACT_CATBOOST), format="cbm"
        )

        # This is the one-off cost we are moving off the request path. `fit` on an
        # in-context learner just installs the context; no gradient step is taken.
        log.info("Fitting TabPFN context (%d rows) at load time", len(tfm_context["x"]))
        model_tfm = build_regressor(workload="serving")
        model_tfm.fit(tfm_context["x"], tfm_context["y"])
        log.info("TabPFN context ready")

        self._predictor = WarmPredictor(
            model_tfm=model_tfm,
            tfm_columns=tfm_context["columns"],
            cb_model=cb_model,
            all_clients=all_clients,
        )

    def predict(self, context, model_input, params=None) -> list[dict[str, Any]]:
        """Accepts a dict, a list of dicts, or a DataFrame of users."""
        users = _normalise_input(model_input)
        return [self._predictor.rank(user) for user in users]


def _normalise_input(model_input) -> list[dict[str, Any]]:
    if isinstance(model_input, pd.DataFrame):
        return model_input.to_dict(orient="records")
    if isinstance(model_input, dict):
        return [model_input]
    if isinstance(model_input, list):
        return list(model_input)
    raise TypeError(f"Unsupported model input type: {type(model_input)!r}")
