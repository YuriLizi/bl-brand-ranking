"""Chart the concurrency sweep against REAL demand.

The sweep alone is misleading as a picture. Its y-axis spans 1s to 146s, so on a linear
scale the entire realistic operating region is squashed into the bottom 1% of the plot --
the chart ends up describing a load level that never occurs and hiding the one that does.

So: log y-axis (1s and 146s both legible), and the region where real traffic actually
lives is shaded. The busiest single second in 62 days of data held 11 arrivals; everything
to the right of that is stress-testing, not forecasting.

Panel 2 is throughput. It is flat, which is the entire capacity argument in one line: past
saturation the service does not go faster, so added concurrency is paid for purely in
waiting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
BLUE_LIGHT = "#cde2fb"
ORANGE = "#eb6834"
BRICK = "#b23b32"
GREEN = "#2f7d5f"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.dpi": 140,
})

ABANDON_SECONDS = 3.0
# Busiest single second across 2025-12-11 -> 2026-02-11 (62,724 registered sessions).
REAL_PEAK_CONCURRENCY = 11


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _crossing(conc: list[int], values: list[float], threshold: float) -> float | None:
    """Interpolate where the curve crosses `threshold`, rather than snapping to a level."""
    for i in range(1, len(conc)):
        if values[i - 1] <= threshold < values[i]:
            span = values[i] - values[i - 1]
            if span <= 0:
                return conc[i]
            frac = (threshold - values[i - 1]) / span
            return conc[i - 1] + frac * (conc[i] - conc[i - 1])
    return None


def render(results: list[dict], out: Path) -> Path:
    results = sorted(results, key=lambda r: r["concurrency"])
    conc = [r["concurrency"] for r in results]
    p50 = [r["latency_ms"]["p50"] / 1000 for r in results]
    p95 = [r["latency_ms"]["p95"] / 1000 for r in results]
    p99 = [r["latency_ms"]["p99"] / 1000 for r in results]
    rps = [r["throughput_rps"] for r in results]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 8.5), height_ratios=[2.4, 1], sharex=True
    )

    # --- real-demand region -----------------------------------------------------------
    ax1.axvspan(0, REAL_PEAK_CONCURRENCY, color=BLUE_LIGHT, alpha=0.55, zorder=0)
    ax1.text(
        REAL_PEAK_CONCURRENCY - 0.6, 115,
        "real demand\nlives here",
        color=BLUE_DARK, fontsize=9.5, ha="right", va="top", fontweight="bold",
    )
    ax1.text(
        REAL_PEAK_CONCURRENCY + 1.6, 115,
        "stress test — never observed in 62 days of data",
        color=INK_MUTED, fontsize=9, ha="left", va="top",
    )

    ax1.axhline(ABANDON_SECONDS, color=BRICK, linewidth=1.2, linestyle="--", zorder=3)
    ax1.text(
        100, ABANDON_SECONDS * 1.12, "users abandon above ~3 s",
        color=BRICK, fontsize=9, va="bottom", ha="right",
    )

    ax1.plot(conc, p99, color=BRICK, linewidth=1.5, marker="o", markersize=3.5,
             label="p99 — worst 1 in 100", zorder=5)
    ax1.plot(conc, p95, color=ORANGE, linewidth=1.1, marker="o", markersize=2.8,
             alpha=0.75, label="p95 — worst 1 in 20", zorder=4)
    ax1.plot(conc, p50, color=BLUE_DARK, linewidth=2.4, marker="o", markersize=4.5,
             label="p50 — typical user", zorder=6)

    ax1.set_yscale("log")
    ax1.set_ylim(0.8, 260)
    ax1.set_yticks([1, 3, 10, 30, 60, 120, 240])
    ax1.set_yticklabels(["1 s", "3 s", "10 s", "30 s", "1 min", "2 min", "4 min"])
    ax1.set_ylabel("wait (log scale)")
    ax1.set_xlim(0, 103)
    ax1.set_title(
        "What one user waits, as simultaneous users pile up\n"
        "measured: 1 container, 1 worker, 5 CPUs, CPU-only TabPFN — 1,015 requests, 0 failures",
        loc="left", pad=12,
    )
    ax1.legend(frameon=False, loc="lower right", ncol=1, fontsize=9,
               bbox_to_anchor=(1.0, 0.02))
    _style(ax1)

    crossed = _crossing(conc, p50, ABANDON_SECONDS)
    if crossed:
        ax1.annotate(
            f"sustained load crosses 3 s\nat only ~{crossed:.0f} concurrent",
            xy=(crossed, ABANDON_SECONDS), xytext=(20, 1.15),
            color=INK_MUTED, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=0.9),
        )

    # --- throughput -------------------------------------------------------------------
    ax2.axvspan(0, REAL_PEAK_CONCURRENCY, color=BLUE_LIGHT, alpha=0.55, zorder=0)
    ax2.plot(conc, rps, color=BLUE, linewidth=2.2, marker="o", markersize=4, zorder=5)
    ax2.axhline(float(np.mean(rps)), color=INK_MUTED, linewidth=0.9, linestyle=":", zorder=3)
    ax2.text(
        100, float(np.mean(rps)) * 1.06, f"mean {np.mean(rps):.2f} rps",
        color=INK_MUTED, fontsize=9, ha="right", va="bottom",
    )
    ax2.set_ylim(0, max(rps) * 1.5)
    ax2.set_xlim(0, 103)
    ax2.set_xlabel("concurrent users")
    ax2.set_ylabel("requests / second")
    ax2.set_title(
        "Throughput never rises — the service saturates at ~1 rps, so extra load becomes pure waiting",
        loc="left", pad=8, fontsize=10.5,
    )
    _style(ax2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "loadtest" / "results" / "sweep.json")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "loadtest" / "charts" / "latency_vs_concurrency.png"
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    print(f"Wrote {render(results, args.out)}")

    print()
    print(f"{'users':>6} {'p50':>9} {'p95':>9} {'p99':>9} {'rps':>7} {'fail':>5}")
    for r in sorted(results, key=lambda x: x["concurrency"]):
        lat = r["latency_ms"]
        print(
            f"{r['concurrency']:>6} {lat['p50']/1000:>8.2f}s {lat['p95']/1000:>8.2f}s "
            f"{lat['p99']/1000:>8.2f}s {r['throughput_rps']:>7.2f} {r['failed']:>5}"
        )


if __name__ == "__main__":
    main()
