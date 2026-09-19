"""Request/response contracts for the ranking endpoint."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from bl_ranker.example_user import EXAMPLE_USER


class UserPayload(BaseModel):
    """One user's post-funnel data, exactly as the funnel emits it.

    Validation is intentionally permissive: the funnel is upstream of us and adding a new
    survey field should not take the endpoint down. Unknown keys are accepted and ignored
    by the preprocessing, which selects the columns it needs by name.
    """

    # The example is attached to the SCHEMA, which is what makes the /docs "Try it out"
    # button work. Without it Swagger pre-fills every field with the literal "string",
    # and pressing Execute returns a 500: "string" is not a timestamp and cellphone is
    # null, so the researcher's preprocessing throws. An API explorer whose default body
    # cannot succeed is worse than none.
    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        json_schema_extra={"example": EXAMPLE_USER},
    )

    session_dt: str
    register_date: str
    conversion_dt: str | None = None
    campaign_id: Any = None
    page: str | None = None
    auto_city: str | None = None
    auto_country: str | None = None
    auto_state: str | None = None
    device_type: str | None = None
    sub1: Any = None
    sub2: Any = None
    sub3: Any = None
    business_type: str | None = None
    credit_score: str | None = None
    industry: str | None = None
    loan_amount: str | None = None
    loan_reason: str | None = None
    monthly_revenue: str | None = None
    time_in_business: str | None = None
    fname: str | None = None
    lname: str | None = None
    cellphone: Any = None


class BrandRank(BaseModel):
    rank: float
    expected_payout: float


class RankResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ranking": {
                    "xlt": {"rank": 1.0, "expected_payout": 64.44},
                    "businessloans.com": {"rank": 2.0, "expected_payout": 25.18},
                    "fundera / nerdwallet": {"rank": 3.0, "expected_payout": 7.26},
                },
                "model_version": "4",
                "latency_ms": 999.56,
            }
        }
    )

    ranking: dict[str, BrandRank] = Field(
        description="Brand name -> rank and expected payout, ordered by expected payout."
    )
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_uri: str
    model_version: str
    tabpfn_execution: str
