"""Chart the open-loop burst runs: does the backlog drain, or grow?

One picture, one question: when 10 users arrive at once and the pattern repeats, does the
Nth burst wait longer than the first?

A flat line means the service drains each burst before the next arrives. A rising line
means it is falling behind and every later user pays for the users ahead of them. That
distinction is invisible in a closed-loop test, which is why this chart exists separately
from `latency_vs_concurrency.png`.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "loadtest" / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"
BLUE_DARK = "#184f95"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BRICK = "#b23b32"
GREEN = "#2f7d5f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_MUTED, "text.color": INK, "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "figure.dpi": 140,
})

SERIES = [
    ("burst_local_6s.json",   "local — burst every 6 s (1.67 rps)",  BRICK,     "-",  2.4),
    ("burst_local_10s.json",  "local — burst every 10 s (1.0 rps)",  ORANGE,    "-",  2.0),
    ("burst_local_12s.json",  "local — burst every 12 s (0.83 rps)", BLUE_DARK, "-",  2.0),
    ("burst_hosted_6s.json",  "hosted — burst every 6 s",            GREEN,     "--", 2.0),
    ("burst_hosted_10s.json", "hosted — burst every 10 s",           GREEN,     ":",  2.2),
]


def per_burst_p50(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    by: dict[int, list[float]] = {}
    for r in data["results"]:
        if r["ok"]:
            by.setdefault(r["burst"], []).append(r["ms"] / 1000)
    return [statistics.median(by[b]) for b in sorted(by)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "loadtest" / "charts" / "burst_backlog.png")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(3.0, color=BRICK, linewidth=1.1, linestyle="--", zorder=2)
    ax.text(10.4, 3.4, "users abandon above ~3 s", color=BRICK, fontsize=9,
            va="bottom", ha="right")

    for fname, label, colour, style, width in SERIES:
        path = args.results / fname
        if not path.exists():
            print(f"  (missing {fname}, skipped)")
            continue
        values = per_burst_p50(path)
        ax.plot(range(len(values)), values, color=colour, linestyle=style, linewidth=width,
                marker="o", markersize=4, label=label, zorder=5)

    ax.set_xlabel("burst number (10 users arrive at once, each time)")
    ax.set_ylabel("median wait, seconds")
    ax.set_title(
        "Does the queue drain between bursts?\n"
        "rising = falling behind; flat = keeping up",
        loc="left", pad=12,
    )
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_ylim(0, None)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
