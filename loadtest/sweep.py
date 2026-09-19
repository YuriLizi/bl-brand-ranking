"""Concurrency sweep: how long does ONE user wait while N users are being served?

This answers the question the briefing implies but does not spell out. "Median and tail
latency" at a single load level is not actionable on its own -- the number a product owner
needs is how the wait a real person experiences degrades as simultaneous users pile up.

Two properties of this script matter:

  * Results are written after EVERY level, not at the end. High concurrency is where a
    service falls over (OOM, connection exhaustion), and a crash at level 10 must not
    destroy the nine levels already measured.
  * The client timeout scales with concurrency. In a closed-loop test a request waits
    behind everyone queued ahead of it, so a fixed timeout eventually measures the
    timeout instead of the service.

Container memory is sampled after each level, because the ceiling here is expected to be
memory (each in-flight request allocates for a TabPFN forward pass), not CPU.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path

from loadtest.simulate import REPO_ROOT, build_payloads, run_load

DEFAULT_LEVELS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def container_memory(service: str = "serving") -> dict | None:
    """Current memory use of the serving container, or None if unavailable."""
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "-q", service],
            capture_output=True, text=True, timeout=30, cwd=REPO_ROOT,
        )
        cid = out.stdout.strip().splitlines()
        if not cid:
            return None
        stats = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}|{{.MemPerc}}|{{.CPUPerc}}", cid[0]],
            capture_output=True, text=True, timeout=60,
        )
        raw = stats.stdout.strip()
        if not raw:
            return None
        mem, perc, cpu = raw.split("|")
        return {"mem_usage": mem.strip(), "mem_percent": perc.strip(), "cpu_percent": cpu.strip()}
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8088/rank")
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "Task" / "bl_full_data.csv")
    parser.add_argument("--pool", type=int, default=500)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "loadtest" / "results" / "sweep.json")
    # Enough requests that each virtual user issues at least ~1.5, with a floor so the low
    # levels still have a usable sample for percentiles.
    parser.add_argument("--min-requests", type=int, default=60)
    parser.add_argument("--per-worker", type=float, default=1.5)
    args = parser.parse_args()

    print(f"Sampling {args.pool} real sessions ...")
    payloads = build_payloads(args.csv, args.pool)
    print(f"Built {len(payloads)} payloads\n")

    results: list[dict] = []
    for level in args.levels:
        total = max(args.min_requests, int(level * args.per_worker))
        # Assume the service sustains >=0.5 rps, then double it, then add slack.
        timeout = max(60.0, (level / 0.5) * 2 + 60)
        print(f"--- concurrency={level}  requests={total}  timeout={timeout:.0f}s ---", flush=True)

        result = asyncio.run(run_load(args.url, payloads, total, level, timeout))
        result["memory_after"] = container_memory()
        results.append(result)

        lat = result["latency_ms"]
        print(
            f"  ok={result['succeeded']} fail={result['failed']} "
            f"rps={result['throughput_rps']} wall={result['wall_seconds']}s"
        )
        print(
            f"  p50={lat['p50']}ms  p90={lat['p90']}ms  p95={lat['p95']}ms  "
            f"p99={lat['p99']}ms  max={lat['max']}ms"
        )
        if result["memory_after"]:
            m = result["memory_after"]
            print(f"  container: mem={m['mem_usage']} ({m['mem_percent']}) cpu={m['cpu_percent']}")
        if result["sample_errors"]:
            print(f"  errors: {result['sample_errors'][:2]}")
        print(flush=True)

        # Write after every level, so a crash at high concurrency keeps what we measured.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
