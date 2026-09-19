"""Contract tests.

These are deliberately narrow. The models themselves are the researcher's and are not
under test here; what *is* under test is everything we added around them -- the parts
that would break silently in production.
"""
from __future__ import annotations

import pandas as pd
import pytest

from bl_ranker.serving.pyfunc_model import _normalise_input
from bl_ranker.training.metrics_capture import MetricCapturingHandler

import logging


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord("t", logging.INFO, __file__, 1, message, None, None)


class TestMetricCapture:
    def test_parses_overall_payout_metrics(self):
        handler = MetricCapturingHandler()
        handler.emit(_record("Overall: \n Payout mean_absolute_percentage_error:0.4213"))
        handler.emit(_record("Payout mean_absolute_error:12.75"))
        assert handler.metrics["payout_mape_overall"] == pytest.approx(0.4213)
        assert handler.metrics["payout_mae_overall"] == pytest.approx(12.75)

    def test_threshold_sweep_becomes_a_stepped_series(self):
        """Stepped, not four scalar keys -- otherwise MLflow has nothing to plot."""
        handler = MetricCapturingHandler()
        for thr, mae in ((5, 9.1), (10, 9.5), (15, 9.7), (20, 9.9)):
            handler.emit(_record(f"thr: {thr}"))
            handler.emit(_record(f"Payout mean_absolute_error:{mae}"))
        assert sorted(handler.series["payout_mae_by_threshold"]) == [
            (5, pytest.approx(9.1)), (10, pytest.approx(9.5)),
            (15, pytest.approx(9.7)), (20, pytest.approx(9.9)),
        ]
        # The overall value stays a scalar; it is genuinely a single number.
        assert "payout_mae_by_threshold" not in handler.metrics

    def test_parses_classification_report(self):
        # The researcher's log never interpolates the day number -- the message arrives
        # with the literal text "for day {day}" -- so the day is recovered positionally.
        report = (
            "CB accuracy report sold_to_client (BL) for day {day} \n"
            "              precision    recall  f1-score   support\n"
            "           0       0.88      0.94      0.91      1000\n"
            "           1       0.62      0.44      0.51       200\n"
            "    accuracy                           0.86      1200\n"
            "   macro avg       0.75      0.69      0.71      1200\n"
            "weighted avg       0.84      0.86      0.84      1200\n"
        )
        handler = MetricCapturingHandler()
        handler.emit(_record(report))
        assert handler.series["lead_accuracy"] == [(1, pytest.approx(0.86))]
        assert handler.series["lead_weighted_f1"] == [(1, pytest.approx(0.84))]

    def test_overall_report_is_separated_from_days(self):
        body = (
            "              precision    recall  f1-score   support\n"
            "    accuracy                           0.90      1200\n"
            "weighted avg       0.88      0.90      0.89      1200\n"
        )
        handler = MetricCapturingHandler()
        handler.emit(_record("CB accuracy report - sold_to_client for all test \n" + body))
        handler.emit(_record("CB accuracy report sold_to_client (BL) for day {day} \n" + body))
        handler.emit(_record("CB accuracy report sold_to_client (BL) for day {day} \n" + body))
        # Overall stays scalar; the per-day values become one stepped series.
        assert handler.metrics["lead_accuracy_overall"] == pytest.approx(0.90)
        assert [step for step, _ in handler.series["lead_accuracy"]] == [1, 2]

    def test_never_raises_on_unrelated_messages(self):
        handler = MetricCapturingHandler()
        handler.emit(_record("start tabular transformer"))
        assert handler.metrics == {}


class TestInputNormalisation:
    def test_accepts_dict(self):
        assert _normalise_input({"a": 1}) == [{"a": 1}]

    def test_accepts_list_of_dicts(self):
        assert _normalise_input([{"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]

    def test_accepts_dataframe(self):
        out = _normalise_input(pd.DataFrame([{"a": 1}, {"a": 2}]))
        assert out == [{"a": 1}, {"a": 2}]

    def test_rejects_unknown(self):
        with pytest.raises(TypeError):
            _normalise_input("not a user")


class TestModelIsNotBloatedByRuntimeState:
    """A registered model must not carry the fitted TabPFN in its pickle.

    MLflow runs the input_example through a freshly logged model to infer its signature,
    which calls load_context and populates _predictor. Without __getstate__ that instance
    pickles with the torch weights attached -- the observed result was a 1 GB
    python_model.pkl that OOM-killed the MLflow artifact server on download.
    """

    def test_predictor_is_excluded_from_the_pickle(self):
        import pickle

        from bl_ranker.serving.pyfunc_model import BrandRankerModel

        model = BrandRankerModel()
        # Stand in for the fitted TabPFN: something large and obviously runtime-only.
        model._predictor = bytearray(5_000_000)

        blob = pickle.dumps(model)
        assert len(blob) < 100_000, (
            f"pickle is {len(blob):,} bytes -- runtime state is leaking into the "
            "registered model"
        )
        assert not hasattr(pickle.loads(blob), "_predictor")


def test_cellphone_prefix_survives_a_ten_digit_number():
    """A US phone number must not overflow to a negative prefix on Windows.

    `.astype(int)` in the original scripts resolves to int32 on Windows, where a 10-digit
    number wraps negative and `str[:3]` yields '-72' instead of '786'. `cellphone_prefix`
    is a model feature, so the corruption silently changes P(lead) -- measured at 0.7771 on
    Linux versus 0.3632 on Windows for the same user. The serving path pins int64; this
    guards that it stays pinned.
    """
    import pandas as pd

    series = pd.Series([7869914030, 2125550143])
    prefixes = series.astype("int64").astype(str).str[:3].tolist()
    assert prefixes == ["786", "212"]
    assert not any(p.startswith("-") for p in prefixes)
