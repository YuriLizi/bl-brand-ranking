"""Replay the REAL arrival pattern through a simulated server.

A concurrency sweep answers "what if N users hammer the service continuously?". Real
traffic does not behave like that: it arrives in bursts and then goes quiet, and a queue
that builds during a burst drains during the lull. Sustained-load numbers therefore
OVERSTATE what a user actually waits.

This replays all 62,724 real `register_date` timestamps through an M/D/c queue with the
measured service time, FIFO, and reports the wait distribution a real user would have
experienced over the whole 62-day period.

Timestamps have 1-second resolution, so arrivals inside one second are modelled two ways:
  * spread   -- uniformly distributed across the second (realistic)
  * burst    -- all at the second boundary (worst case, conservative)
Both are reported, because the gap between them is the honest uncertainty band.

SERVICE TIME IS PER CONFIGURATION, NOT A CONSTANT
-------------------------------------------------
Each configuration is given as `servers:service_seconds`, and BOTH halves must come from a
measurement of that configuration. Sweeping several server counts against one shared
service time is the obvious thing to do and it is wrong here: it assumes a worker runs as
fast alongside the others as it does alone, i.e. that the box has unlimited cores.

On this hardware it does not. One request already parallelises across every core, so a
second worker does not add compute, it splits it. Measured:

    1 worker  x 5 threads -> 1.07 rps  =>  0.93 s per request
    2 workers x 2 threads -> 0.83 rps  =>  2.41 s per request  (each worker SLOWER)

An earlier version of this script took a single `--service` and a list of `--servers`, and
produced a table in which 2 workers halved p99 and cut the over-3s rate. That was an
artifact of the shared-service-time assumption, and it directly contradicted the measured
throughput regression. Fed the real numbers, the same replay says 2 workers are worse on
p50, p95, p99 AND the over-3s rate. Hence this interface: a configuration cannot be
expressed without stating the service time measured for it.
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def simulate(arrivals: np.ndarray, service: float, servers: int) -> dict:
    """FIFO M/D/c queue. Returns response-time percentiles in seconds."""
    free = [0.0] * servers
    heapq.heapify(free)
    waits = np.empty(len(arrivals))
    for i, t in enumerate(arrivals):
        earliest = heapq.heappop(free)
        start = t if t > earliest else earliest
        waits[i] = start - t
        heapq.heappush(free, start + service)
    response = waits + service
    return {
        "queued_pct": float((waits > 0.001).mean() * 100),
        "p50": float(np.percentile(response, 50)),
        "p95": float(np.percentile(response, 95)),
        "p99": float(np.percentile(response, 99)),
        "p999": float(np.percentile(response, 99.9)),
        "max": float(response.max()),
        "over_3s_pct": float((response > 3).mean() * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "Task" / "bl_full_data.csv")
    # Each entry is "servers:service_seconds". BOTH halves must come from a measurement of
    # THAT configuration -- see the module docstring for why a shared service time across a
    # list of server counts produces a wrong answer on this hardware.
    parser.add_argument(
        "--config", nargs="+", default=["1:0.93", "2:2.41"],
        help='configurations as servers:service_seconds, e.g. "1:0.93" "2:2.41"',
    )
    args = parser.parse_args()

    configs = []
    for entry in args.config:
        servers, _, service = entry.partition(":")
        if not service:
            parser.error(f"--config entry {entry!r} must be servers:service_seconds")
        configs.append((int(servers), float(service)))

    df = pd.read_csv(args.csv, low_memory=False)
    ts = pd.to_datetime(
        df.loc[df["register_date"].notna(), "register_date"], errors="coerce"
    ).dropna().sort_values()
    base = ts.min()
    seconds = (ts - base).dt.total_seconds().to_numpy()

    rng = np.random.default_rng(42)
    spread = np.sort(seconds + rng.random(len(seconds)))

    print(f"Replaying {len(seconds):,} real arrivals over {seconds.max()/86400:.0f} days")
    print()

    for name, arrivals in (("burst (worst case)", seconds), ("spread (realistic)", spread)):
        print(f"=== arrivals within a second modelled as: {name} ===")
        header = (
            f"{'workers':>8} {'service':>9} {'queued':>8} {'p50':>8} {'p95':>8} "
            f"{'p99':>8} {'p99.9':>8} {'max':>9} {'>3s':>7}"
        )
        print(header)
        for c, service in configs:
            r = simulate(arrivals, service, c)
            print(
                f"{c:>8} {service:>8.2f}s {r['queued_pct']:>7.1f}% {r['p50']:>7.2f}s "
                f"{r['p95']:>7.2f}s {r['p99']:>7.2f}s {r['p999']:>7.2f}s "
                f"{r['max']:>8.2f}s {r['over_3s_pct']:>6.2f}%"
            )
        print()


if __name__ == "__main__":
    main()
