"""Registry alias management -- promotion and rollback.

A rollback plan that only exists in a README is not a rollback plan. MLflow has no CLI
command for alias management (`mlflow models` covers serving and packaging only), so this
module is the operational interface:

    python -m bl_ranker.rollback --list                 # what is registered, what is live
    python -m bl_ranker.rollback --to-previous          # swap champion <-> previous
    python -m bl_ranker.rollback --to-version 3         # pin champion to an exact version

Rolling back does not rebuild or retrain anything: it moves an alias and the serving
process picks up the change on its next start. The previous champion is always already
tagged `@previous` by the training run that replaced it, so the target is never in doubt
at the moment you need it.
"""
from __future__ import annotations

import argparse
import logging
import sys

import mlflow
from mlflow.tracking import MlflowClient

from bl_ranker.config import get_settings

log = logging.getLogger(__name__)


def _client() -> MlflowClient:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def _alias_version(client: MlflowClient, name: str, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(name, alias).version
    except Exception:
        return None


def list_versions() -> None:
    settings = get_settings()
    client = _client()
    name = settings.registered_model_name

    champion = _alias_version(client, name, settings.champion_alias)
    previous = _alias_version(client, name, settings.previous_alias)

    versions = sorted(
        client.search_model_versions(f"name='{name}'"), key=lambda m: int(m.version)
    )
    if not versions:
        print(f"No versions registered for '{name}'.")
        return

    print(f"Registered model: {name}")
    print(f"{'version':>8}  {'alias':<12} {'status':<10} {'run':<10} created")
    for mv in versions:
        marks = []
        if mv.version == champion:
            marks.append(f"@{settings.champion_alias}")
        if mv.version == previous:
            marks.append(f"@{settings.previous_alias}")
        alias_text = ",".join(marks) or "-"
        print(
            f"{mv.version:>8}  {alias_text:<12} {mv.status:<10} "
            f"{mv.run_id[:8]:<10} {mv.creation_timestamp}"
        )
    print(f"\nServing resolves: {settings.resolved_model_uri}")


def set_champion(version: str) -> None:
    settings = get_settings()
    client = _client()
    name = settings.registered_model_name

    current = _alias_version(client, name, settings.champion_alias)
    if current == version:
        print(f"Version {version} is already @{settings.champion_alias}; nothing to do.")
        return

    # Demote the incumbent first, so @previous always points at what was just replaced.
    if current is not None:
        client.set_registered_model_alias(name, settings.previous_alias, current)
        print(f"v{current} -> @{settings.previous_alias}")

    client.set_registered_model_alias(name, settings.champion_alias, version)
    print(f"v{version} -> @{settings.champion_alias}")
    print("\nRestart the serving process for the change to take effect:")
    print("  docker compose restart serving")


def to_previous() -> None:
    settings = get_settings()
    client = _client()
    previous = _alias_version(client, settings.registered_model_name, settings.previous_alias)
    if previous is None:
        print(
            f"No @{settings.previous_alias} alias is set -- there is nothing to roll back "
            "to. This is expected when only one version has ever been promoted."
        )
        sys.exit(1)
    set_champion(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="show versions and live aliases")
    group.add_argument(
        "--to-previous", action="store_true", help="roll back to the @previous version"
    )
    group.add_argument("--to-version", metavar="N", help="pin @champion to an exact version")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    if args.list:
        list_versions()
    elif args.to_previous:
        to_previous()
    else:
        set_champion(str(args.to_version))


if __name__ == "__main__":
    main()
