"""Alias promotion / rollback logic, against a fake registry.

The rollback path is the one piece of operational code you use precisely when something is
already going wrong, so it should not be the piece that has never been tested. These tests
pin the invariant that matters: after any promotion or rollback, `@champion` and
`@previous` point at different versions and neither is left dangling.
"""
from __future__ import annotations

import pytest

from bl_ranker import rollback


class FakeClient:
    """Minimal stand-in for MlflowClient's alias surface."""

    def __init__(self, aliases: dict[str, str] | None = None):
        self.aliases: dict[str, str] = dict(aliases or {})
        self.calls: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name: str, alias: str):
        if alias not in self.aliases:
            raise Exception(f"alias {alias} not found")
        return type("MV", (), {"version": self.aliases[alias]})()

    def set_registered_model_alias(self, name: str, alias: str, version: str):
        self.aliases[alias] = version
        self.calls.append((alias, version))


@pytest.fixture
def fake(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(rollback, "_client", lambda: client)
    return client


def test_first_promotion_sets_champion_only(fake, capsys):
    rollback.set_champion("1")
    assert fake.aliases == {"champion": "1"}


def test_promotion_demotes_incumbent(fake):
    fake.aliases = {"champion": "1"}
    rollback.set_champion("2")
    assert fake.aliases["champion"] == "2"
    assert fake.aliases["previous"] == "1"


def test_rollback_swaps_the_two_aliases(fake):
    fake.aliases = {"champion": "2", "previous": "1"}
    rollback.to_previous()
    assert fake.aliases["champion"] == "1"
    assert fake.aliases["previous"] == "2"


def test_rollback_is_reversible(fake):
    fake.aliases = {"champion": "2", "previous": "1"}
    rollback.to_previous()
    rollback.set_champion("2")
    assert fake.aliases["champion"] == "2"
    assert fake.aliases["previous"] == "1"


def test_aliases_never_collide(fake):
    """The invariant that makes a rollback safe to run under pressure."""
    fake.aliases = {"champion": "1"}
    for version in ("2", "3", "4"):
        rollback.set_champion(version)
        assert fake.aliases["champion"] != fake.aliases["previous"]


def test_promoting_the_live_version_is_a_no_op(fake, capsys):
    fake.aliases = {"champion": "2", "previous": "1"}
    rollback.set_champion("2")
    assert fake.calls == []
    assert "already" in capsys.readouterr().out


def test_rollback_without_previous_exits_nonzero(fake):
    fake.aliases = {"champion": "1"}
    with pytest.raises(SystemExit) as exc:
        rollback.to_previous()
    assert exc.value.code == 1
