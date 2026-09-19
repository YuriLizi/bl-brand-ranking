"""Load simulation against the ranking endpoint.

Reports median and tail latency, which is what the briefing asks for. Two details matter
for the numbers to mean anything:

  * We measure **client-observed** latency (the number a user's browser would feel),
    not just the server's internal timer, so queueing shows up.
  * Requests are drawn from real rows of the dataset rather than replaying one fixed
    payload, so caches are exercised the way production would exercise them.

Open-loop vs closed-loop: this is a closed-loop test with a fixed number of concurrent
virtual users. At the concurrency levels the funnel actually produces that is the honest
model; a note on where it under-reports is in the README.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

SURVEY_COLUMNS = [
    "session_dt", "conversion_dt", "register_date", "campaign_id", "page",
    "auto_city", "auto_country", "auto_state", "device_type", "sub1", "sub2", "sub3",
    "business_type", "credit_score", "industry", "loan_amount", "loan_reason",
    "monthly_revenue", "time_in_business", "fname", "lname", "cellphone",
]


def build_payloads(csv_path: Path, n: int, seed: int = 42) -> list[dict]:
    """Sample real, complete sessions to use as request bodies."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df["register_date"].notna() & df["cellphone"].notna() & df["fname"].notna()]
    df = df.dropna(subset=["credit_score", "industry", "loan_amount", "monthly_revenue"])
    df = df[[c for c in SURVEY_COLUMNS if c in df.columns]]
    sample = df.sample(n=min(n, len(df)), random_state=seed)

    payloads = []
    for record in sample.to_dict(orient="records"):
        clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        clean["cellphone"] = int(float(clean["cellphone"]))
        payloads.append(clean)
    random.Random(seed).shuffle(payloads)
    return payloads


async def _worker(
    client: httpx.AsyncClient, url: str, payloads: list[dict], counter: list[int],
    total: int, latencies: list[float], errors: list[str], timeout: float = 30.0,
) -> None:
    while True:
        index = counter[0]
        if index >= total:
            return
        counter[0] += 1
        payload = payloads[index % len(payloads)]
        started = time.perf_counter()
        try:
            response = await client.post(url, json=payload, timeout=timeout)
            elapsed = (time.perf_counter() - started) * 1000
            if response.status_code == 200:
                latencies.append(elapsed)
            else:
                errors.append(f"{response.status_code}: {response.text[:120]}")
        except Exception as exc:
            errors.append(repr(exc))


async def run_load(
    url: str, payloads: list[dict], total: int, concurrency: int, timeout: float = 30.0
) -> dict:
    latencies: list[float] = []
    errors: list[str] = []
    counter = [0]

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        wall_start = time.perf_counter()
        await asyncio.gather(
            *(
                _worker(client, url, payloads, counter, total, latencies, errors, timeout)
                for _ in range(concurrency)
            )
        )
        wall = time.perf_counter() - wall_start

    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return float("nan")
        index = min(len(latencies) - 1, int(round(p / 100 * len(latencies))) - 1)
        return latencies[max(0, index)]

    return {
        "requests": total,
        "concurrency": concurrency,
        "succeeded": len(latencies),
        "failed": len(errors),
        "wall_seconds": round(wall, 2),
        "throughput_rps": round(len(latencies) / wall, 2) if wall else 0.0,
        "latency_ms": {
            "min": round(latencies[0], 1) if latencies else None,
            "p50": round(statistics.median(latencies), 1) if latencies else None,
            "p90": round(pct(90), 1),
            "p95": round(pct(95), 1),
            "p99": round(pct(99), 1),
            "max": round(latencies[-1], 1) if latencies else None,
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
        },
        "sample_errors": errors[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test the /rank endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/rank")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "Task" / "bl_full_data.csv")
    parser.add_argument("--pool", type=int, default=200, help="distinct payloads to sample")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "loadtest" / "results" / "results.json")
    # Per-request client timeout. MUST scale with concurrency: in a closed-loop test a
    # request waits behind everyone ahead of it, so latency grows ~linearly with
    # concurrency. At concurrency 100 against a ~1 rps service a request waits ~90s, and
    # a 30s timeout would record that as a FAILURE rather than as slow -- turning a
    # capacity measurement into a measurement of the timeout.
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    print(f"Sampling {args.pool} real sessions from {args.csv} ...")
    payloads = build_payloads(args.csv, args.pool)
    print(f"Built {len(payloads)} payloads\n")

    results = []
    for concurrency in args.concurrency:
        print(f"--- concurrency={concurrency}, requests={args.requests} ---")
        result = asyncio.run(
            run_load(args.url, payloads, args.requests, concurrency, args.timeout)
        )
        results.append(result)
        latency = result["latency_ms"]
        print(
            f"  ok={result['succeeded']} fail={result['failed']} "
            f"rps={result['throughput_rps']}\n"
            f"  p50={latency['p50']}ms  p90={latency['p90']}ms  "
            f"p95={latency['p95']}ms  p99={latency['p99']}ms  max={latency['max']}ms"
        )
        if result["sample_errors"]:
            print(f"  errors: {result['sample_errors']}")
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
