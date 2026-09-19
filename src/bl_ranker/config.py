"""Central configuration.

Every environment-specific value lives here so that the *same* code runs against a
local Docker Compose stack and against Databricks. Switching platforms is a matter of
environment variables, never a code change.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------ data
    raw_csv_path: Path = Field(default=REPO_ROOT / "Task" / "bl_full_data.csv")
    delta_table_path: Path = Field(default=REPO_ROOT / "data" / "delta" / "bl_sessions")
    # On Databricks this becomes a Unity Catalog name, e.g. "main.bl.sessions".
    delta_table_uri: str | None = None

    # --------------------------------------------------------------- mlflow
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment: str = "bl_brand_ranking"
    registered_model_name: str = "bl_brand_ranker"
    champion_alias: str = "champion"
    previous_alias: str = "previous"

    # -------------------------------------------------------------- serving
    serving_host: str = "0.0.0.0"
    serving_port: int = 8000
    # Threads torch may use *per worker*. With N uvicorn workers on a 6-core box,
    # leaving this at 1-2 avoids oversubscription; see README "Latency tuning".
    torch_num_threads: int = 2
    model_uri: str | None = None  # defaults to models:/<name>@<champion>

    # ------------------------------------------------------------ scheduler
    schedule_cron: str = "0 5 * * 0"  # Sunday 05:00, as required by the briefing
    schedule_timezone: str = "UTC"

    # ------------------------------------------------------------- training
    days_for_test: int = 7
    random_seed: int = 42

    # --------------------------------------------------------------- tabpfn
    # Size of TabPFN's internal ensemble. 8 is the library default and therefore what the
    # researcher's code uses implicitly. This is the ONLY remaining latency lever that
    # changes predictions -- see README "Latency vs fidelity".
    tabpfn_n_estimators: int = 8

    @property
    def resolved_model_uri(self) -> str:
        if self.model_uri:
            return self.model_uri
        return f"models:/{self.registered_model_name}@{self.champion_alias}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
