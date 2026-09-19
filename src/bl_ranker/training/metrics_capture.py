"""Turn the researcher's free-text log into comparable MLflow metrics.

The briefing asks for two different things and we satisfy both without touching the
researcher's evaluation code:

  * "Save researcher-defined log as an artifact for accuracy assessment" -> the .log file
    is uploaded verbatim, so a human can read the full per-day classification reports.
  * "Log technical parameters to MLflow so runs are comparable" -> the numbers that
    actually matter for comparison are also parsed out and logged as MLflow *metrics*,
    because you cannot chart or sort runs by a text file.

We attach a logging handler rather than editing the evaluation methods, so the parsing is
purely additive.
"""
from __future__ import annotations

import logging
import re

_MAPE = re.compile(r"Payout mean_absolute_percentage_error:\s*([0-9.eE+-]+)")
_MAE = re.compile(r"Payout mean_absolute_error:\s*([0-9.eE+-]+)")
_THR = re.compile(r"^thr:\s*(\d+)")
# "  accuracy    0.87  1234" / "weighted avg 0.81 0.87 0.83 1234"
_ACCURACY = re.compile(r"^\s*accuracy\s+([0-9.]+)\s+(\d+)\s*$", re.MULTILINE)
_WEIGHTED_F1 = re.compile(r"^\s*weighted avg\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)\s+\d+", re.MULTILINE)
_ALL_TEST = re.compile(r"for all test")


class MetricCapturingHandler(logging.Handler):
    """Collects metric-shaped values emitted by the researcher's logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.metrics: dict[str, float] = {}
        self._current_threshold: int | None = None
        # The researcher's per-day reports are emitted in order but are NOT labelled with
        # the day: bl_models_train.py:270 builds the message as
        #     f'CB accuracy report ' + target_col + ' (BL) for day {day} ...'
        # where the segment containing {day} is a plain concatenated string, not an
        # f-string, so it reaches the log as the literal text "for day {day}". We
        # therefore recover the day positionally -- the report containing "for all test"
        # is the overall one, and each report after it is the next day in sequence.
        self._day_counter = 0
        # Sequential values go here as {name: [(step, value), ...]} so they can be logged
        # with a step and CHARTED. Logging `lead_accuracy_day1..day7` as seven separate
        # scalar keys, which is what this class used to do, gives MLflow nothing to plot:
        # each key holds a single point. One key with seven steps is a line.
        self.series: dict[str, list[tuple[int, float]]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - never break training on a log record
            return

        thr_match = _THR.match(message.strip())
        if thr_match:
            self._current_threshold = int(thr_match.group(1))
            return

        self._capture_payout(message)
        self._capture_classification(message)

    def _capture_payout(self, message: str) -> None:
        for pattern, name in ((_MAPE, "payout_mape"), (_MAE, "payout_mae")):
            match = pattern.search(message)
            if not match:
                continue
            value = float(match.group(1))
            # The whole sweep is ONE curve: step 0 is the unrestricted figure, then one
            # point per threshold. Logging the unrestricted value as its own scalar as
            # well produced a separate single-bar chart in the MLflow UI that said
            # nothing the curve does not already show.
            step = 0 if self._current_threshold is None else self._current_threshold
            self.series.setdefault(f"{name}_by_threshold", []).append((step, value))
            if self._current_threshold is None:
                # Kept as a scalar too: this is the headline number for comparing runs
                # in the table, where a series cannot be sorted on.
                self.metrics[f"{name}_overall"] = value

    def _capture_classification(self, message: str) -> None:
        if "CB accuracy report" not in message:
            return
        overall = bool(_ALL_TEST.search(message))
        if not overall:
            self._day_counter += 1
        day = self._day_counter

        accuracy = _ACCURACY.search(message)
        weighted_f1 = _WEIGHTED_F1.search(message)

        if overall:
            if accuracy:
                self.metrics["lead_accuracy_overall"] = float(accuracy.group(1))
            if weighted_f1:
                self.metrics["lead_weighted_f1_overall"] = float(weighted_f1.group(1))
            return

        # Per-day values are a time series across the held-out week.
        if accuracy:
            self.series.setdefault("lead_accuracy", []).append((day, float(accuracy.group(1))))
            # Support alongside it. Without this a day holding ONE row plots at 1.00 and
            # reads exactly like a day holding 1,900 rows. In this dataset days 3, 4 and 7
            # hold 23, 3 and 1 rows -- the source data has a near-outage on 2026-02-08
            # (2 registered sessions all day), and the last bucket spans only the instant
            # up to max_date. Charting accuracy without support makes noise look like
            # perfect performance.
            self.series.setdefault("lead_support", []).append((day, float(accuracy.group(2))))
        if weighted_f1:
            self.series.setdefault("lead_weighted_f1", []).append(
                (day, float(weighted_f1.group(1)))
            )
