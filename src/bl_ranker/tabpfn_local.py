"""Local, in-process TabPFN.

WHY THIS EXISTS
---------------
The researcher's scripts reach TabPFN through `tabpfn-client`, which is a thin client for
Prior Labs' *hosted* API. That design puts two network round-trips on a synchronous,
user-facing request:

  1. `TabPFNRegressor().fit(context)` uploads the 1000-row in-context training set;
  2. `.predict(rows)` sends the query rows and waits for the forward pass.

TabPFN is an in-context learner, so `fit` is not training -- it is just handing the model
its context. That means the *same* model is available as open-source weights, which
priorlabs.ai lists as a first-class deployment option alongside the API and VPC options.

Running it locally removes the network from the request path entirely and, more
importantly, lets us fit the context ONCE at process start instead of once per request.
The public API is identical (`fit` / `predict`), so the researcher's call sites are
untouched -- only the class they resolve to changes.

Trade-off, stated plainly: the open-source weights are TabPFN v2, while the hosted client
is v3.0. Predictions are close but not bit-identical. That is a deliberate,
documented choice, and `USE_HOSTED_TABPFN=true` restores the original behaviour.

THE TWO WORKLOADS NEED OPPOSITE SETTINGS
----------------------------------------
Measured on a Ryzen 5 5600X (CPU-only, context = 1000 rows):

  fit_mode="fit_preprocessors" (default)
      Cost is dominated by re-processing the context on every call, so it is nearly
      independent of how many rows you ask for: 1 row ~8.2s, 200 rows ~8.6s.
      Optimal for BATCH scoring, where you have thousands of rows in one call.

  fit_mode="fit_with_cache"
      Precomputes the context's transformer state at fit time. Per-call cost becomes
      roughly linear in query rows (~110 ms/row), and the one-off fit costs ~92s.
      Optimal for ONLINE serving, where every request is exactly 10 rows (one per brand)
      and the fit happens once at process start, off the request path.

Picking the wrong one is expensive in both directions, which is why this is a parameter
rather than a constant. Verified: the two modes agree to within float32 noise
(max abs diff 3.7e-4 on payouts around $148) and produce identical ranking order, so this
is a pure compute choice with no effect on model semantics.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

import torch

log = logging.getLogger(__name__)

Workload = Literal["batch", "serving"]


def configure_torch_threads(num_threads: int) -> None:
    """Pin torch's intra-op parallelism.

    With several uvicorn workers on one box, letting each torch instance grab every core
    causes oversubscription and makes tail latency much worse than the median. One or two
    threads per worker is consistently better under concurrency.
    """
    torch.set_num_threads(max(1, num_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Can only be set before any parallel work starts; harmless if already running.
        pass


def build_regressor(
    workload: Workload = "serving",
    n_estimators: int | None = None,
    use_hosted: bool | None = None,
):
    """Return a TabPFNRegressor -- local by default, hosted only if explicitly asked.

    Args:
        workload: "serving" uses the cached context (fast per small request);
            "batch" uses the default mode (fast for large single calls).
        n_estimators: size of TabPFN's internal ensemble. The library default is 8, which
            is what the researcher's code uses implicitly. Lowering it is the only
            remaining latency lever and it *does* change predictions.
        use_hosted: force the hosted API. Defaults to the USE_HOSTED_TABPFN env var.
    """
    if use_hosted is None:
        use_hosted = os.environ.get("USE_HOSTED_TABPFN", "").lower() in {"1", "true", "yes"}

    if use_hosted:
        from tabpfn_client import TabPFNRegressor, set_access_token

        token = os.environ.get("TABPFN_TOKEN")
        if not token:
            raise RuntimeError(
                "USE_HOSTED_TABPFN is set but TABPFN_TOKEN is missing. Either provide the "
                "token or unset USE_HOSTED_TABPFN to use the local open-source weights."
            )
        set_access_token(token)
        log.info("TabPFN: using HOSTED client (network call per request)")
        return TabPFNRegressor(ignore_pretraining_limits=True)

    from tabpfn import TabPFNRegressor

    from bl_ranker.config import get_settings

    settings = get_settings()
    estimators = n_estimators if n_estimators is not None else settings.tabpfn_n_estimators
    fit_mode = "fit_with_cache" if workload == "serving" else "fit_preprocessors"

    log.info(
        "TabPFN: local open-source weights on CPU (workload=%s, fit_mode=%s, n_estimators=%d)",
        workload,
        fit_mode,
        estimators,
    )
    return TabPFNRegressor(
        ignore_pretraining_limits=True,
        device="cpu",
        n_estimators=estimators,
        fit_mode=fit_mode,
        random_state=settings.random_seed,
    )
