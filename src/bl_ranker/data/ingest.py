"""CSV -> Delta ingestion.

The briefing's original README asked for the data to live in a *table* rather than a
loose CSV. We use Delta Lake through `delta-rs`, which needs no Spark and no JVM, so the
identical table format works on a laptop and on Databricks Unity Catalog.

The reason this matters for MLOps: a Delta table has a transaction log, so every
training run can record the exact table *version* it consumed. Combined with model
versioning in the MLflow registry, that makes a run reproducible from both sides --
you can answer "what model was serving, and what data produced it" for any point in time.
"""
from __future__ import annotations

import logging

import pandas as pd

from bl_ranker.config import get_settings

log = logging.getLogger(__name__)

# Columns the CSV carries twice (business_name appears in two positions); pandas
# de-duplicates them as business_name / business_name.1 on read.
_DUPLICATE_SUFFIXED = ".1"


def _delta_rs():
    """Import delta-rs lazily.

    The two backends are mutually exclusive and neither should be a hard import: on
    Databricks, Delta is reached through Spark and `deltalake` is not installed at all,
    so importing it at module level makes the package unimportable there. The reverse
    holds locally, where there is no SparkSession. Each is imported only on the path that
    actually uses it.
    """
    from deltalake import DeltaTable, write_deltalake

    return DeltaTable, write_deltalake


def _spark():
    """Return an active SparkSession, or None when running off-cluster.

    Local runs use delta-rs and never touch Spark. On Databricks there is always a
    session available, and Unity Catalog tables are only reachable through it.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        return None
    try:
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:  # pragma: no cover - no cluster available
        return None


def _require_spark(uri: str):
    spark = _spark()
    if spark is None:
        raise RuntimeError(
            f"DELTA_TABLE_URI is set to '{uri}', which is a Unity Catalog table, but no "
            "SparkSession is available. Either run this on Databricks, or unset "
            "DELTA_TABLE_URI to use the local Delta table at DELTA_TABLE_PATH."
        )
    return spark


def ingest_csv_to_delta(mode: str = "overwrite") -> int:
    """Load the raw CSV into the Delta table. Returns the resulting version.

    Writes to a Unity Catalog table when DELTA_TABLE_URI is set (Databricks), and to a
    local delta-rs table otherwise. The two paths produce the same logical table, which
    is what lets the identical training code run in both places.
    """
    settings = get_settings()
    csv_path = settings.raw_csv_path
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. Download bl_full_data.csv from the "
            "Drive link in the briefing and place it there, or set RAW_CSV_PATH."
        )

    log.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop the unnamed index column the export carried, and any duplicate-name column,
    # so the table has a clean schema. Neither is used by the model.
    df = df.loc[:, [c for c in df.columns if not c.startswith("Unnamed")]]
    df = df.loc[:, [c for c in df.columns if not c.endswith(_DUPLICATE_SUFFIXED)]]

    # Delta needs a stable schema; the raw export mixes types in several columns.
    # Everything the model reads is re-typed inside the researcher's own preprocessing,
    # so storing as string here is lossless with respect to the model.
    df = df.astype(str).where(df.notna(), None)

    if settings.delta_table_uri:
        uri = settings.delta_table_uri
        spark = _require_spark(uri)
        log.info("Writing %d rows to Unity Catalog table %s (mode=%s)", len(df), uri, mode)
        (
            spark.createDataFrame(df)
            .write.format("delta")
            .mode(mode)
            .option("overwriteSchema", "true")
            .saveAsTable(uri)
        )
        version = current_version()
        log.info("Table %s now at version %d", uri, version)
        return version

    DeltaTable, write_deltalake = _delta_rs()
    target = str(settings.delta_table_path)
    settings.delta_table_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Writing %d rows to Delta table at %s (mode=%s)", len(df), target, mode)
    write_deltalake(target, df, mode=mode, schema_mode="overwrite")

    version = DeltaTable(target).version()
    log.info("Delta table now at version %d", version)
    return version


def read_delta(version: int | None = None) -> pd.DataFrame:
    """Read the Delta table (optionally a historical version) as a pandas DataFrame."""
    settings = get_settings()

    if settings.delta_table_uri:
        uri = settings.delta_table_uri
        spark = _require_spark(uri)
        reader = spark.read
        if version is not None:
            # Delta time travel: pin training to the exact snapshot that was recorded.
            reader = reader.option("versionAsOf", version)
        return reader.table(uri).toPandas()

    DeltaTable, _ = _delta_rs()
    target = str(settings.delta_table_path)
    dt = DeltaTable(target, version=version) if version is not None else DeltaTable(target)
    return dt.to_pandas()


def current_version() -> int:
    """Latest committed version of the table, on either backend."""
    settings = get_settings()

    if settings.delta_table_uri:
        uri = settings.delta_table_uri
        spark = _require_spark(uri)
        row = spark.sql(f"DESCRIBE HISTORY {uri} LIMIT 1").select("version").collect()
        return int(row[0][0]) if row else 0

    DeltaTable, _ = _delta_rs()
    return DeltaTable(str(settings.delta_table_path)).version()


def main() -> None:
    """CLI entrypoint.

    Serverless Databricks jobs have no cluster on which to set environment variables, so
    the Databricks-specific configuration is passed as arguments instead and exported
    before settings are first constructed.
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Load bl_full_data.csv into the Delta table.")
    parser.add_argument(
        "--table",
        default=None,
        help="Unity Catalog table, e.g. main.bl.bl_sessions. Omit to use the local table.",
    )
    parser.add_argument("--csv", default=None, help="Override the source CSV path.")
    args = parser.parse_args()

    if args.table:
        os.environ["DELTA_TABLE_URI"] = args.table
    if args.csv:
        os.environ["RAW_CSV_PATH"] = args.csv

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    version = ingest_csv_to_delta()
    print(f"delta_version={version}")


if __name__ == "__main__":
    main()
