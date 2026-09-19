"""FastAPI ranking service.

WHY FASTAPI RATHER THAN `mlflow models serve`
---------------------------------------------
MLflow's built-in scoring server is convenient but it is a generic DataFrame-in /
DataFrame-out wrapper: it adds a serialisation layer we do not need, gives no control over
warm-up, and its request shape does not match the dictionary contract the funnel expects.

We keep the *registry* (versioning, aliases, rollback) and drop the *scoring server*: the
app loads the `@champion` pyfunc at startup and serves it behind a thin, typed HTTP layer.
That is the combination that gives both governance and latency.

Everything expensive -- weights, the TabPFN context fit, the NameDataset -- happens once in
the lifespan startup hook. A request does preprocessing, one CatBoost `predict_proba` and
one TabPFN forward pass over 10 rows, and nothing else.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, ORJSONResponse

from bl_ranker.config import get_settings
from bl_ranker.example_user import EXAMPLE_USER as _WARMUP_USER
from bl_ranker.serving.schemas import HealthResponse, RankResponse, UserPayload

log = logging.getLogger(__name__)

STATE: dict[str, Any] = {"model": None, "version": "unknown", "brands": None}


def _resolve_version(settings) -> str:
    try:
        from mlflow.tracking import MlflowClient

        version = MlflowClient().get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        return version.version
    except Exception:
        return "unknown"


def _brand_universe_size() -> int | None:
    """How many brands the champion scores against, for the landing page.

    Read through MLflow's wrapper internals deliberately. The registered model bundles its
    own copy of the serving code, so adding an accessor to `pyfunc_model.py` would not
    affect a version that is already registered -- this has to work against the code that
    shipped with the artifact. Guarded accordingly: if the layout changes, the landing page
    shows a dash and the endpoint is unaffected.
    """
    try:
        predictor = STATE["model"]._model_impl.python_model._predictor
        return int(len(predictor._all_clients))
    except Exception:
        log.debug("Could not determine brand universe size", exc_info=True)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    uri = settings.resolved_model_uri
    log.info("Loading model from %s", uri)
    started = time.perf_counter()
    STATE["model"] = mlflow.pyfunc.load_model(uri)
    STATE["version"] = _resolve_version(settings)
    log.info(
        "Model loaded in %.1fs (version=%s)", time.perf_counter() - started, STATE["version"]
    )

    # Warm the whole path once so the first real user does not pay for lazy imports,
    # CatBoost's first-call allocation or torch kernel selection.
    try:
        STATE["model"].predict([_WARMUP_USER])
        log.info("Warm-up prediction complete")
        STATE["brands"] = _brand_universe_size()
    except Exception:
        log.exception("Warm-up prediction failed -- serving anyway")

    yield
    STATE.clear()


app = FastAPI(
    title="BL Brand Ranking",
    version="1.0.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Landing page. Without it, clicking the port returns a bare 404."""
    from bl_ranker.serving.index_page import render

    hosted = os.environ.get("USE_HOSTED_TABPFN", "").lower() in {"1", "true", "yes"}
    return HTMLResponse(
        render(
            version=STATE.get("version", "unknown"),
            loaded=STATE.get("model") is not None,
            tabpfn="hosted" if hosted else "local",
            model_uri=get_settings().resolved_model_uri,
            brands=STATE.get("brands"),
            example_user=_WARMUP_USER,
        )
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Readiness and live model version",
)
async def health() -> HealthResponse:
    """Report whether the service is ready to rank, and what it is serving.

    Returns `200` once the model is loaded and warmed, and `503` while it is still
    starting -- a cold start takes ~130s, so an orchestrator should allow for that before
    considering the container unhealthy.

    `model_version` is the registry version currently behind the `@champion` alias, which
    is what a rollback changes.
    """
    settings = get_settings()
    if STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    hosted = os.environ.get("USE_HOSTED_TABPFN", "").lower() in {"1", "true", "yes"}
    return HealthResponse(
        status="ok",
        model_uri=settings.resolved_model_uri,
        model_version=STATE["version"],
        tabpfn_execution="hosted-api" if hosted else "local-oss-cpu",
    )


# MAINTAINER NOTE -- this handler is deliberately `def`, not `async def`.
#
# The work here is ~1s of blocking CPU: a CatBoost `predict_proba` and a TabPFN forward
# pass. FastAPI runs an `async def` handler directly on the event loop, so that second of
# compute blocks the loop entirely -- protocol handling, new connections and every other
# in-flight request stall behind it. A plain `def` handler is dispatched to a threadpool
# instead, leaving the loop free.
#
# This was not visible at low load. It appeared only at concurrency 8 over 60 requests, as
# 4 dropped connections:
#
#     RemoteProtocolError('Server disconnected without sending a response.')
#
# with p99 blowing out to 14s. Smaller runs (16 requests) passed cleanly, which is a good
# argument for load-testing at a realistic sample size rather than a token one.
#
# Keep this as a comment, not a docstring: FastAPI publishes the docstring as the
# endpoint's description in /docs, where internal rationale is noise to the caller.
@app.post(
    "/rank",
    response_model=RankResponse,
    summary="Rank brands for one user",
)
def rank(payload: UserPayload) -> RankResponse:
    """Rank the brand universe for a single user by expected payout.

    Send one user's post-funnel data as a JSON object. Unknown fields are accepted and
    ignored, so you can post a raw session row unchanged.

    Returns every brand scored for this user, ordered best-first, each with its
    `expected_payout` and its 1-based `rank`.

    A user who cannot be scored -- for example one with too many missing survey answers --
    returns an empty `ranking` rather than an error. A user with no `register_date` cannot
    be a lead at all and returns `422`.
    """
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    started = time.perf_counter()
    try:
        result = model.predict([payload.model_dump()])[0]
    except ValueError as exc:
        # Raised when the user cannot be a lead (no register_date) -- a client error.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        log.exception("Ranking failed")
        raise HTTPException(status_code=500, detail="ranking failed") from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    return RankResponse(
        ranking=_as_ranking(result),
        model_version=STATE["version"],
        latency_ms=round(elapsed_ms, 2),
    )


def _as_ranking(result: dict) -> dict:
    """Normalise the researcher's two return shapes into one.

    `predict_` returns the brand dictionary normally, but returns the sentinel
    `{"expected_payout": 0, "prob_lead": 0}` when preprocessing drops the row (for
    example too many missing survey answers). Both mean "no ranking to show"; the
    endpoint reports that as an empty ranking rather than a 500.
    """
    if not result:
        return {}
    if all(isinstance(value, dict) for value in result.values()):
        return result
    return {}
