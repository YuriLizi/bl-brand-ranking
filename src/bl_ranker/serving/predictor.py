"""Warm, service-shaped predictor.

The researcher's `BLPayoutModelsPredict` is written as a one-shot script: it builds a new
log file, re-reads `all_clients.csv` from disk and re-fits TabPFN on *every* call. That is
correct for a notebook and unusable on a synchronous path where a user is waiting.

This subclass keeps all of the feature engineering inherited and unchanged, and fixes only
the service-level concerns:

  1. Models are loaded and TabPFN's context is fitted ONCE, at process start.
  2. `all_clients.csv` is read once and cached.
  3. Logging uses a normal module logger instead of creating a log file per request.
  4. The gender lookup is memoised. The brand cross-join happens *before* feature
     engineering, so the original calls `NameDataset.search()` once per brand for the same
     first name -- 10 identical lookups per request. Memoising returns identical values.
  5. A real bug is fixed: `import_preprocess` reads the module-level global `user_data`
     instead of `self.user_data` (original line 58). That only works because `__main__`
     happens to define that global; inside a service it raises NameError.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bl_ranker.original.bl_exp_payout_predictor import BLPayoutModelsPredict

log = logging.getLogger(__name__)


class WarmPredictor(BLPayoutModelsPredict):
    """A long-lived predictor: construct once, call `rank()` per request."""

    def __init__(self, model_tfm, tfm_columns, cb_model, all_clients: pd.DataFrame):
        # Deliberately does NOT call super().__init__: that would build a per-instance
        # log file and clear the root logger's handlers.
        self._model_tfm = model_tfm
        self._tfm_columns = tfm_columns
        self._cb_model = cb_model
        self._all_clients = all_clients
        self.predictors_path = ""
        self.user_data: dict[str, Any] = {}
        self.logger = log
        self._gender_cache: dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------- overrides
    def load_models(self):
        """Already warm -- no disk read, no network, no TabPFN re-fit."""
        return self._model_tfm, self._tfm_columns, self._cb_model

    def detect_gender_with_confidence(self, fname):
        """Memoised delegate to the researcher's own implementation.

        The brand cross-join happens *before* feature engineering, so the original runs
        this lookup once per brand for the same first name -- 10 identical NameDataset
        searches per request. We call the parent's method and cache its result, so the
        values are identical by construction; only the redundant work disappears.
        """
        cached = self._gender_cache.get(fname)
        if cached is None:
            cached = super().detect_gender_with_confidence(fname)
            self._gender_cache[fname] = cached
        return cached

    def import_preprocess(self):
        """Parent's body with three fixes: `self.user_data` (was a global), a cached
        client list (was a per-request `read_csv`) and an int64 cast on the phone number
        (see below). Everything else is verbatim."""
        bl_data = pd.DataFrame([self.user_data])
        all_clients = self._all_clients
        needed_columns = [
            "session_dt", "conversion_dt", "register_date",
            "campaign_id", "page", "auto_city", "auto_country", "auto_state",
            "device_type", "sub1", "sub2", "sub3",
            "business_type", "credit_score", "industry", "loan_amount", "loan_reason",
            "monthly_revenue", "time_in_business", "fname", "lname", "cellphone",
        ]
        bl_data = bl_data[needed_columns]
        if bl_data["register_date"].isna().any():
            raise ValueError("user cannot be a lead - register_date is absent")

        bl_data = bl_data.merge(all_clients, how="cross")
        bl_data = bl_data[bl_data["client_name"] != "other"]
        bl_data = bl_data.rename(
            columns={"auto_city": "city", "auto_state": "state", "auto_country": "country"}
        )
        bl_data[["country", "state", "city", "sub1", "sub2", "sub3"]] = bl_data[
            ["country", "state", "city", "sub1", "sub2", "sub3"]
        ].fillna("Other")
        bl_data["country_state"] = np.where(
            bl_data["country"] == "United States", bl_data["state"], bl_data["country"]
        )
        bl_data["session_dt"] = pd.to_datetime(bl_data["session_dt"], errors="coerce")
        bl_data["register_date"] = pd.to_datetime(bl_data["register_date"], errors="coerce")
        bl_data["sub1"] = bl_data["sub1"].astype(str)
        bl_data["sub2"] = bl_data["sub2"].astype(str)
        bl_data["sub3"] = bl_data["sub3"].astype(str)
        # The original is `.astype(int)`, which takes numpy's PLATFORM DEFAULT: int64 on
        # Linux, int32 on Windows. A US phone number has 10 digits (~7.8e9) and int32 tops
        # out at 2,147,483,647, so on Windows the value wraps negative and the prefix comes
        # out wrong -- silently:
        #
        #     7869914030 -> .astype(int) -> -720020562 -> str[:3] -> '-72'   (want '786')
        #
        # `cellphone_prefix` is a CatBoost feature, so this is not cosmetic. A model trained
        # on Linux has never seen '-72'; it arrives as an unknown category encoded as zeros
        # and P(lead) for the example user moves from 0.7771 to 0.3632 -- enough to change
        # both the scores and how many brands clear the 0.01 cut-off.
        #
        # int64 is pinned explicitly so serving agrees with training on every platform.
        # Training always runs on Linux (Docker, Databricks), so this aligns Windows with
        # the values the model was actually fitted on rather than inventing new behaviour.
        # The same line exists in both original scripts and is reported, not edited, there.
        bl_data["cellphone_prefix"] = (
            bl_data["cellphone"].astype("int64").astype(str).str[:3].astype(str)
        )
        return bl_data

    # ------------------------------------------------------------- public API
    def rank(self, user_data: dict[str, Any]) -> dict[str, dict[str, float]]:
        """Score one user against every brand and return the ranked dictionary."""
        self.user_data = user_data
        return self.predict_()
