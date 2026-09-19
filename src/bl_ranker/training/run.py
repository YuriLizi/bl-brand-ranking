"""Training entrypoint -- both modes, fully tracked in MLflow.

    python -m bl_ranker.training.run --mode train_test   # evaluate, write no artifacts
    python -m bl_ranker.training.run --mode production   # train on all data, register

Mode semantics are exactly the researcher's `train_test` flag; this module only adds
tracking, artifact registration and alias management around them.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from bl_ranker.config import get_settings
from bl_ranker.data.ingest import current_version
from bl_ranker.example_user import EXAMPLE_USER
from bl_ranker.serving.pyfunc_model import (
    ARTIFACT_ALL_CLIENTS,
    ARTIFACT_CATBOOST,
    ARTIFACT_TFM_CONTEXT,
    BrandRankerModel,
)
from bl_ranker.training.metrics_capture import MetricCapturingHandler
from bl_ranker.training.trainer import ProductionTrainer

log = logging.getLogger(__name__)

SRC_ROOT = Path(__file__).resolve().parents[1]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SRC_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _technical_params(mode: str, delta_version: int, rows: int, settings) -> dict[str, object]:
    """Everything needed to make two runs comparable and to reproduce either one."""
    return {
        "mode": mode,
        "delta_table_version": delta_version,
        "input_rows": rows,
        "days_for_test": settings.days_for_test,
        "random_seed": settings.random_seed,
        # CatBoost hyper-parameters, mirrored from the researcher's code so that a change
        # there shows up as a diff between runs rather than silently.
        "catboost_depth": 8,
        "catboost_n_estimators": 800,
        "catboost_eval_metric": "F1",
        "catboost_task_type": "CPU",
        # TabPFN
        "tabpfn_context_size": 1000,
        "tabpfn_execution": "local-oss-cpu",
        "git_sha": _git_sha(),
    }


def run(mode: str, delta_version: int | None = None, register: bool = True) -> str:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    train_test = mode == "train_test"
    workdir = Path(tempfile.mkdtemp(prefix="bl_train_"))
    resolved_version = delta_version if delta_version is not None else current_version()

    run_name = f"{mode}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    with mlflow.start_run(run_name=run_name) as active_run:
        trainer = ProductionTrainer(
            output_predictors_path=str(workdir) + "/",
            train_test=train_test,
            delta_version=resolved_version,
        )

        # Attach after the trainer's constructor, which resets the root logger's handlers.
        capture = MetricCapturingHandler()
        logging.getLogger().addHandler(capture)

        mlflow.log_params(
            _technical_params(mode, resolved_version, trainer.snapshot_rows, settings)
        )

        started = datetime.now(timezone.utc)
        try:
            trainer.fit_()
        finally:
            logging.getLogger().removeHandler(capture)
            for handler in logging.getLogger().handlers:
                handler.flush()

        # Duration is operational, not model quality. As a metric it earns its own
        # (meaningless) single-bar chart in the UI; as a tag it stays visible on the run
        # and in the table without adding noise to the charts.
        duration = (datetime.now(timezone.utc) - started).total_seconds()
        mlflow.set_tag("training_seconds", f"{duration:.1f}")
        if capture.metrics:
            mlflow.log_metrics(capture.metrics)
            log.info("Logged %d scalar metrics", len(capture.metrics))

        # Sequential values are logged WITH A STEP, which is what makes MLflow draw a
        # line instead of a dot. Seven per-day accuracies under one key is a trend across
        # the held-out week; seven separate keys is seven unplottable scalars.
        for name, points in capture.series.items():
            for step, value in sorted(points):
                mlflow.log_metric(name, value, step=step)
            log.info("Logged series '%s' with %d steps", name, len(points))

        _log_train_metrics(trainer, capture)
        _log_run_summary(trainer, capture, workdir)

        # The researcher-defined log, verbatim, for accuracy assessment.
        log_file = trainer.log_file
        if log_file and log_file.exists():
            mlflow.log_artifact(str(log_file), artifact_path="researcher_log")

        _log_charts(trainer, workdir)

        if train_test:
            mlflow.set_tag("writes_artifacts", "false")
            log.info("train_test mode complete -- no artifacts registered by design")
            return active_run.info.run_id

        _register(workdir, settings, active_run.info.run_id, register)
        return active_run.info.run_id


def _log_train_metrics(trainer, capture) -> None:
    """Log the same classifier metrics on the TRAIN set, and the gap between them.

    The researcher's evaluation reports the held-out week only, so a run cannot show
    whether the model generalises or has memorised. These three additions make that
    legible without touching their code:

        lead_accuracy_train        same metric, training data
        lead_weighted_f1_train
        lead_overfit_gap           train accuracy - test accuracy

    The gap is the one to watch. Near zero means the held-out numbers are trustworthy; a
    large positive gap means the model fits the training week far better than the next
    one, and the weekly retrain is shipping memorisation. It is also the natural thing to
    gate a promotion on.
    """
    captured = getattr(trainer, "captured", {})
    y_true = captured.get("train_y_true")
    y_pred = captured.get("train_y_pred")
    if y_true is None or y_pred is None:
        return

    try:
        from sklearn.metrics import accuracy_score, f1_score

        train_accuracy = float(accuracy_score(y_true, y_pred))
        train_f1 = float(f1_score(y_true, y_pred, average="weighted"))
        mlflow.log_metric("lead_accuracy_train", train_accuracy)
        mlflow.log_metric("lead_weighted_f1_train", train_f1)

        test_accuracy = capture.metrics.get("lead_accuracy_overall")
        if test_accuracy is not None:
            gap = train_accuracy - float(test_accuracy)
            mlflow.log_metric("lead_overfit_gap", gap)
            log.info(
                "Train accuracy %.4f vs test %.4f (gap %+.4f)",
                train_accuracy, test_accuracy, gap,
            )
        else:
            log.info("Train accuracy %.4f (no test figure in this mode)", train_accuracy)
    except Exception:
        log.warning("Could not log train-set metrics", exc_info=True)


def _log_run_summary(trainer, capture, workdir: Path) -> None:
    """One chart that carries what a page of single-bar scalar charts cannot.

    MLflow draws every scalar metric as its own bar. For a single run that bar has nothing
    to compare against, so a screen of them says very little. This puts train beside test,
    and the per-day accuracy beside the number of rows behind each point.
    """
    captured = getattr(trainer, "captured", {})
    y_true = captured.get("train_y_true")
    y_pred = captured.get("train_y_pred")
    if y_true is None or y_pred is None:
        return

    try:
        from sklearn.metrics import accuracy_score, f1_score

        from bl_ranker.training import plots

        pairs = []
        test_acc = capture.metrics.get("lead_accuracy_overall")
        test_f1 = capture.metrics.get("lead_weighted_f1_overall")
        if test_acc is not None:
            pairs.append(("accuracy", float(accuracy_score(y_true, y_pred)), float(test_acc)))
        if test_f1 is not None:
            pairs.append(
                ("weighted F1", float(f1_score(y_true, y_pred, average="weighted")),
                 float(test_f1))
            )
        if not pairs:
            return

        support = dict(capture.series.get("lead_support", []))
        per_day = [
            (day, value, support.get(day, 0))
            for day, value in sorted(capture.series.get("lead_accuracy", []))
        ]

        chart_dir = workdir / "charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        path = plots.run_summary(pairs, per_day, chart_dir)
        mlflow.log_artifact(str(path), artifact_path="charts")
        log.info("Logged run summary chart")
    except Exception:
        log.warning("Could not produce the run summary chart", exc_info=True)


def _log_charts(trainer, workdir: Path) -> None:
    """Log the charts MLflow renders inline on the Artifacts tab.

    Every one is best-effort: a plotting failure must never lose a training run that has
    already cost minutes of CatBoost and TabPFN time.
    """
    # Imported inside the guard: charts are a nice-to-have, and a missing plotting stack
    # must never destroy a run that has already spent minutes on CatBoost and TabPFN.
    try:
        from bl_ranker.training import plots
    except Exception:
        log.warning("Plotting unavailable (matplotlib missing?) - skipping charts", exc_info=True)
        return

    chart_dir = workdir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    captured = trainer.captured
    made: list[Path] = []

    model = captured.get("cb_model")
    names = captured.get("feature_names")
    if model is not None and names:
        for fn in (plots.feature_importance, plots.feature_importance_table):
            try:
                made.append(fn(model, names, chart_dir))
            except Exception:
                log.warning("Could not produce %s", fn.__name__, exc_info=True)

    if captured.get("y_true") is not None and captured.get("y_pred") is not None:
        try:
            made.append(
                plots.confusion_matrix(captured["y_true"], captured["y_pred"], chart_dir)
            )
        except Exception:
            log.warning("Could not produce confusion matrix", exc_info=True)

    if captured.get("payout_actual") is not None and captured.get("payout_pred") is not None:
        try:
            made.append(
                plots.payout_pred_vs_actual(
                    captured["payout_actual"], captured["payout_pred"], chart_dir
                )
            )
        except Exception:
            log.warning("Could not produce payout scatter", exc_info=True)

    # Calibration: a genuine within-run series (one point per probability decile), and the
    # diagnostic that bears most directly on ranking quality.
    if captured.get("y_true") is not None and captured.get("y_proba") is not None:
        try:
            path, rows = plots.calibration(captured["y_true"], captured["y_proba"], chart_dir)
            made.append(path)
            for decile, predicted, observed, n in rows:
                mlflow.log_metric("calibration_predicted", predicted, step=decile)
                mlflow.log_metric("calibration_observed", observed, step=decile)
                mlflow.log_metric("calibration_support", n, step=decile)
            if rows:
                # One number for "how far off is P(lead) overall", support-weighted.
                total = sum(r[3] for r in rows)
                ece = sum(abs(r[1] - r[2]) * r[3] for r in rows) / total if total else 0.0
                mlflow.log_metric("calibration_error_expected", ece)
                log.info("Calibration logged over %d deciles (ECE %.4f)", len(rows), ece)
        except Exception:
            log.warning("Could not produce calibration", exc_info=True)

    # Per-brand: the dimension the product is organised around, and the one an overall
    # accuracy hides completely.
    if captured.get("y_true") is not None and captured.get("brand") is not None:
        try:
            png, csv, rows = plots.per_brand_performance(
                captured["y_true"], captured["y_pred"], captured["brand"], chart_dir
            )
            made.extend([png, csv])
            for rank, brand, accuracy, support in rows:
                mlflow.log_metric("lead_accuracy_by_brand", accuracy, step=rank)
                mlflow.log_metric("lead_support_by_brand", support, step=rank)
            # Step numbers carry no names, so record the mapping alongside them.
            mlflow.log_param(
                "brand_rank_order", ", ".join(f"{r}={b}" for r, b, _, _ in rows)[:480]
            )
            log.info("Per-brand performance logged for %d brands", len(rows))
        except Exception:
            log.warning("Could not produce per-brand performance", exc_info=True)

    for path in made:
        mlflow.log_artifact(str(path), artifact_path="charts")
    if made:
        log.info("Logged %d chart artifacts", len(made))


def _register(workdir: Path, settings, run_id: str, register: bool) -> None:
    """Log the composite model and move the champion alias to the new version."""
    artifacts = {
        ARTIFACT_CATBOOST: str(workdir / "CB_bl_lead.cbm"),
        ARTIFACT_TFM_CONTEXT: str(workdir / "payout_tfm_context.joblib"),
        ARTIFACT_ALL_CLIENTS: str(workdir / "all_clients.csv"),
    }
    missing = [name for name, path in artifacts.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Training did not produce expected artifacts: {missing}")

    # The example request is logged as a plain ARTIFACT, deliberately NOT as
    # `input_example=`.
    #
    # Passing input_example makes MLflow infer and enforce a schema from that single
    # record, and the researcher's example happens to carry `sub1` as an int (1121993).
    # Real funnel data carries it as a zero-padded string ('01122005'), so the inferred
    # signature rejected genuine production traffic outright:
    #
    #     Incompatible input types for column sub1. Can not safely convert object to int64
    #
    # Every request in a load test failed with a 500 for that reason. The upstream funnel
    # is loosely typed by nature and the preprocessing already coerces what it needs, so a
    # strict inferred schema buys nothing and breaks real calls. The FastAPI layer does the
    # validation that belongs here, and it is permissive on purpose.
    example_path = workdir / "example_request.json"
    example_path.write_text(json.dumps(EXAMPLE_USER, indent=2), encoding="utf-8")
    mlflow.log_artifact(str(example_path), artifact_path="example")

    # requirements.txt likewise only exists in a checkout. When it is there we pin the
    # model environment to the exact versions we tested; when it is not (an installed
    # wheel, e.g. on Databricks) we let MLflow infer, which is correct because the
    # package's own dependencies are declared in pyproject.toml.
    requirements_file = SRC_ROOT.parents[1] / "requirements.txt"
    pip_requirements = str(requirements_file) if requirements_file.exists() else None

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=BrandRankerModel(),
        artifacts=artifacts,
        code_paths=[str(SRC_ROOT)],
        registered_model_name=settings.registered_model_name if register else None,
        pip_requirements=pip_requirements,
    )
    log.info("Composite pyfunc model logged")

    if not register:
        return

    client = MlflowClient()
    versions = client.search_model_versions(f"name='{settings.registered_model_name}'")
    newest = max(versions, key=lambda v: int(v.version))

    # Demote the current champion to `previous` so a rollback is one command away.
    try:
        incumbent = client.get_model_version_by_alias(
            settings.registered_model_name, settings.champion_alias
        )
        if incumbent.version != newest.version:
            client.set_registered_model_alias(
                settings.registered_model_name, settings.previous_alias, incumbent.version
            )
            log.info("Previous champion (v%s) tagged as @previous", incumbent.version)
    except Exception:
        log.info("No existing champion to demote -- this is the first version")

    client.set_registered_model_alias(
        settings.registered_model_name, settings.champion_alias, newest.version
    )
    log.info(
        "Model version %s promoted to @%s (run %s)",
        newest.version,
        settings.champion_alias,
        run_id,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BL brand-ranking models.")
    parser.add_argument(
        "--mode",
        choices=["train_test", "production"],
        default="train_test",
        help="train_test evaluates and writes nothing; production trains on all data "
        "and registers a new model version.",
    )
    parser.add_argument(
        "--delta-version",
        type=int,
        default=None,
        help="Pin training to a historical Delta table version (default: latest).",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Log the model to the run but do not touch the registry.",
    )
    # Serverless Databricks jobs have no cluster on which to set environment variables,
    # so the platform-specific configuration arrives as arguments and is exported before
    # settings are first constructed.
    parser.add_argument(
        "--table", default=None, help="Unity Catalog table, e.g. main.bl.bl_sessions."
    )
    parser.add_argument(
        "--tracking-uri", default=None, help="MLflow tracking URI ('databricks' on DBX)."
    )
    parser.add_argument(
        "--registry-uri", default=None, help="MLflow registry URI ('databricks-uc' on DBX)."
    )
    parser.add_argument(
        "--registered-model",
        default=None,
        help="Registered model name; must be catalog.schema.name under Unity Catalog.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help=(
            "MLflow experiment. A bare name works against a local tracking server, but "
            "Databricks requires an absolute workspace path, e.g. "
            "/Users/you@example.com/bl_brand_ranking."
        ),
    )
    args = parser.parse_args()

    import os

    if args.table:
        os.environ["DELTA_TABLE_URI"] = args.table
    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    if args.registry_uri:
        os.environ["MLFLOW_REGISTRY_URI"] = args.registry_uri
        mlflow.set_registry_uri(args.registry_uri)
    if args.registered_model:
        os.environ["REGISTERED_MODEL_NAME"] = args.registered_model
    if args.experiment:
        os.environ["MLFLOW_EXPERIMENT"] = args.experiment

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    run_id = run(args.mode, args.delta_version, register=not args.no_register)
    print(json.dumps({"run_id": run_id, "mode": args.mode}))


if __name__ == "__main__":
    main()
