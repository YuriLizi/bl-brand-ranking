"""The example payload must have exactly one source of truth.

`EXAMPLE_USER` lives in the package because the runtime needs it (warm-up call, MLflow
input example) and a repo-relative path does not survive being installed as a wheel.
`examples/user.json` exists purely so a reviewer can `curl -d @examples/user.json`.

Two copies means they drift, and the one a reviewer pastes stops matching the one the
model was registered with. This test makes that impossible.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bl_ranker.example_user import EXAMPLE_USER

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_JSON = REPO_ROOT / "examples" / "user.json"


def test_json_matches_the_packaged_constant():
    if not EXAMPLE_JSON.exists():
        pytest.skip("examples/user.json is absent (installed wheel, not a checkout)")
    assert json.loads(EXAMPLE_JSON.read_text(encoding="utf-8")) == EXAMPLE_USER


def test_example_carries_every_field_the_predictor_requires():
    """The preprocessing selects these by name and raises KeyError on any that is absent."""
    required = {
        "session_dt", "conversion_dt", "register_date", "campaign_id", "page",
        "auto_city", "auto_country", "auto_state", "device_type", "sub1", "sub2", "sub3",
        "business_type", "credit_score", "industry", "loan_amount", "loan_reason",
        "monthly_revenue", "time_in_business", "fname", "lname", "cellphone",
    }
    assert required <= set(EXAMPLE_USER), f"missing: {required - set(EXAMPLE_USER)}"


def test_register_date_is_present():
    """Without it the predictor raises: the user cannot be a lead."""
    assert EXAMPLE_USER.get("register_date")


def test_openapi_example_is_the_real_payload():
    """The /docs "Try it out" body must be a request that actually succeeds.

    Without an example on the schema, Swagger pre-fills every field with the literal
    "string" and pressing Execute returns a 500 -- "string" is not a timestamp and
    cellphone is null, so the preprocessing throws. An API explorer whose default body
    cannot work is worse than not having one.
    """
    from bl_ranker.serving.schemas import UserPayload

    example = UserPayload.model_config.get("json_schema_extra", {}).get("example")
    assert example == EXAMPLE_USER, "schema example must be the canonical request"

    schema = UserPayload.model_json_schema()
    assert "example" in schema, "example did not reach the generated JSON schema"

    # And it must validate against the model it documents.
    assert UserPayload(**example).session_dt == EXAMPLE_USER["session_dt"]
