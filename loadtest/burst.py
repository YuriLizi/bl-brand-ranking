"""Open-loop burst test: N users arrive at once, then a lull, repeatedly.

WHY THIS AND NOT THE SWEEP
--------------------------
`sweep.py` is CLOSED-loop: a fixed pool of virtual users, each of which waits for its reply
before sending again. That model cannot build a backlog -- if the server slows down, the
load offered to it slows down too, which is exactly what real traffic does NOT do.

This is OPEN-loop: bursts are fired on a wall-clock schedule whether or not the previous
burst has finished. If arrivals outpace service, the queue grows and each burst is slower
than the last. That growth is the thing worth seeing, and it is invisible to a closed-loop
test.

It also matches the shape of the real data (§6): the busiest second held 11 arrivals, and
bursts are separated by lulls rather than being sustained.

Read the per-burst table, not the aggregate. A flat column means the system drains each
burst before the next arrives; a rising column means it is falling behind and every
subsequent user pays for the users ahead of them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from loadtest.simulate import REPO_ROOT, build_payloads


async def _one(client, url, payload, burst_index, results, timeout):
    started = time.perf_counter()
    try:
        r = await client.post(url, json=payload, timeout=timeout)
        elapsed = (time.perf_counter() - started) * 1000
        if r.status_code == 200:
            results.append({"burst": burst_index, "ms": elapsed, "ok": True, "err": None})
        else:
            results.append({"burst": burst_index, "ms": elapsed, "ok": False,
                            "err": f"{r.status_code}: {r.text[:100]}"})
    except Exception as exc:
        results.append({"burst": burst_index, "ms": (time.perf_counter() - started) * 1000,
                        "ok": False, "err": repr(exc)[:120]})


async def run_bursts(url, payloads, size, gap, duration, timeout):
    results: list[dict] = []
    tasks: list[asyncio.Task] = []
    limits = httpx.Limits(max_connections=size * 4, max_keepalive_connections=size * 4)
    async with httpx.AsyncClient(limits=limits) as client:
        t0 = time.perf_counter()
        burst_index = 0
        cursor = 0
        while time.perf_counter() - t0 < duration:
            scheduled = time.perf_counter() - t0
            for _ in range(size):
                payload = payloads[cursor % len(payloads)]
                cursor += 1
                # Fire and DO NOT await -- the schedule does not wait for the server.
                tasks.append(asyncio.create_task(
                    _one(client, url, payload, burst_index, results, timeout)
                ))
            print(f"  burst {burst_index:>2} fired at t={scheduled:5.1f}s "
                  f"({size} requests, {len(tasks)} in flight)", flush=True)
            burst_index += 1
            await asyncio.sleep(gap)
        print(f"\n  schedule finished; draining {sum(not t.done() for t in tasks)} "
              f"outstanding requests ...", flush=True)
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0
    return results, wall


def report(results, wall, size, gap, label):
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"\n=== {label} ===")
    print(f"offered load: {size} requests every {gap}s = {size/gap:.2f} rps")
    print(f"sent {len(results)}, ok {len(ok)}, failed {len(bad)}, wall {wall:.1f}s, "
          f"completed {len(ok)/wall:.2f} rps")

    print(f"\n  {'burst':>5} {'n':>3} {'p50':>9} {'max':>9} {'fail':>5}")
    by_burst: dict[int, list[dict]] = {}
    for r in results:
        by_burst.setdefault(r["burst"], []).append(r)
    for b in sorted(by_burst):
        rows = by_burst[b]
        good = [r["ms"] for r in rows if r["ok"]]
        nfail = sum(1 for r in rows if not r["ok"])
        if good:
            print(f"  {b:>5} {len(rows):>3} {statistics.median(good)/1000:>8.2f}s "
                  f"{max(good)/1000:>8.2f}s {nfail:>5}")
        else:
            print(f"  {b:>5} {len(rows):>3} {'--':>9} {'--':>9} {nfail:>5}")

    if ok:
        lat = sorted(r["ms"] for r in ok)
        pct = lambda q: lat[min(len(lat)-1, int(q/100*len(lat)))]
        print(f"\n  overall p50={statistics.median(lat)/1000:.2f}s  "
              f"p95={pct(95)/1000:.2f}s  p99={pct(99)/1000:.2f}s  max={lat[-1]/1000:.2f}s")
    if bad:
        seen: dict[str, int] = {}
        for r in bad:
            key = r["err"].split(":")[0][:60]
            seen[key] = seen.get(key, 0) + 1
        print("\n  failures:")
        for k, v in sorted(seen.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>3}x {k}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8088/rank")
    parser.add_argument("--size", type=int, default=10, help="requests per burst")
    parser.add_argument("--gap", type=float, default=6.0, help="seconds between bursts")
    parser.add_argument("--duration", type=float, default=60.0, help="how long to keep firing")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--pool", type=int, default=500)
    parser.add_argument("--csv", type=Path, default=REPO_ROOT / "Task" / "bl_full_data.csv")
    parser.add_argument("--label", default="burst test")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    payloads = build_payloads(args.csv, args.pool)
    print(f"Built {len(payloads)} payloads")
    print(f"Firing {args.size} requests every {args.gap}s for {args.duration}s "
          f"= {args.size/args.gap:.2f} rps offered\n")

    results, wall = asyncio.run(
        run_bursts(args.url, payloads, args.size, args.gap, args.duration, args.timeout)
    )
    report(results, wall, args.size, args.gap, args.label)

    if args.out:
        args.out.write_text(json.dumps(
            {"size": args.size, "gap": args.gap, "duration": args.duration,
             "wall": wall, "results": results}, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
