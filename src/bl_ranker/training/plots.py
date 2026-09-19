"""Charts logged as MLflow artifacts.

MLflow renders images inline on a run's Artifacts tab, and until now this project logged
none -- so the UI had numbers but nothing to look at.

Three charts, each answering a question a reviewer actually asks:

  * feature importance -- *what is the lead model keying on?* The most revealing of the
    three: it shows whether the classifier leans on the survey answers (signal about the
    business) or on traffic-source fields like `sub1`/`campaign_id` (signal about where
    the ad ran, which will not generalise when campaigns change).
  * confusion matrix -- *how does it fail?* Precision and recall in the log are summary
    statistics; the matrix shows which direction the errors go.
  * predicted vs actual payout -- *is 7.2% MAPE good?* A single number cannot show that
    the tail brands sit in a different regime from `fundera / nerdwallet`.

Colours follow the project palette: one hue for magnitude (blue, light->dark), text in ink
tokens rather than the series colour, recessive grid and axes, no chart junk.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on a training box or a Databricks worker
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)

# --- palette -------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
BLUE_DARK = "#184f95"
BLUE_LIGHT = "#cde2fb"
ORANGE = "#eb6834"
BRICK = "#b23b32"   # flags a value that needs attention, e.g. a wide train/test gap

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


def _finish(ax) -> None:
    """Recessive frame: keep the data, drop the box."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)


def feature_importance(model, feature_names: list[str], out_dir: Path, top_n: int = 20) -> Path:
    """Horizontal bar chart of CatBoost feature importance, most important at the top."""
    values = model.get_feature_importance()
    order = np.argsort(values)[-top_n:]
    names = [feature_names[i] for i in order]
    vals = values[order]

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(names))))
    # One series, one hue, one shade. Bar length already encodes magnitude, so varying
    # lightness would be redundant -- and the light steps fall below the contrast floor
    # against this surface, which is exactly the case the relief rule warns about.
    ax.barh(range(len(names)), vals, color=BLUE, height=0.68)
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlabel("importance")
    ax.set_title(f"What the lead classifier keys on (top {len(names)})", loc="left", pad=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

    # Direct-label only the leaders; a number on every bar is noise.
    for i, v in enumerate(vals):
        if v >= vals.max() * 0.5:
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8.5, color=INK_MUTED)

    _finish(ax)
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_matrix(y_true, y_pred, out_dir: Path) -> Path:
    """2x2 counts for the lead classifier, as a sequential single-hue heatmap."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    labels = ["not sold", "sold"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Lead classifier: where the errors go", loc="left", pad=12)

    total = cm.sum()
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / total if total else 0
            ax.text(
                j, i, f"{cm[i, j]:,}\n{frac:.1%}", ha="center", va="center", fontsize=11,
                color="white" if cm[i, j] > cm.max() * 0.55 else INK,
            )
    # A surface-coloured gap between adjacent fills, so the four cells read as four marks
    # rather than one block.
    ax.set_xticks([0.5], minor=True)
    ax.set_yticks([0.5], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def payout_pred_vs_actual(actual, predicted, out_dir: Path) -> Path:
    """Scatter with a y=x reference. Makes the MAPE legible and exposes the tail."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    hi = float(max(actual.max(), predicted.max())) * 1.05
    # Reference first, so points sit above it.
    ax.plot([0, hi], [0, hi], color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    ax.scatter(
        actual, predicted, s=26, color=BLUE, alpha=0.45,
        edgecolors=SURFACE, linewidths=0.6, zorder=2,
    )
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("actual payout ($)")
    ax.set_ylabel("predicted payout ($)")
    ax.set_title("Payout: predicted vs actual", loc="left", pad=12)
    ax.text(
        hi * 0.97, hi * 0.90, "perfect prediction", ha="right", fontsize=8.5,
        color=INK_MUTED, style="italic",
    )
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(ax)
    fig.tight_layout()
    path = out_dir / "payout_pred_vs_actual.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def feature_importance_table(model, feature_names: list[str], out_dir: Path) -> Path:
    """The same numbers as CSV, so they can be diffed between runs."""
    df = pd.DataFrame(
        {"feature": feature_names, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    path = out_dir / "feature_importance.csv"
    df.to_csv(path, index=False)
    return path


def calibration(y_true, y_proba, out_dir: Path, bins: int = 10):
    """Reliability curve, plus the per-bin numbers for logging as a stepped series.

    This is the diagnostic that matters most for THIS product and is absent from the
    researcher's report. The ranking is `expected_payout = P(lead) x payout`, so a
    systematically over- or under-confident P reorders brands even when every class
    prediction is correct. Accuracy cannot see that; calibration can.

    Returns (path, rows) where rows is [(decile, mean_predicted, observed_rate, n), ...].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(y_proba, edges[1:-1], right=False), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append((b + 1, float(y_proba[mask].mean()), float(y_true[mask].mean()), n))

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=1)
    if rows:
        xs = [r[1] for r in rows]
        ys = [r[2] for r in rows]
        ns = np.array([r[3] for r in rows], dtype=float)
        ax.plot(xs, ys, color=BLUE, linewidth=2, zorder=2)
        # Marker area carries support. The top deciles can hold a handful of rows, and an
        # equal-sized dot there reads as confidently as one backed by thousands -- which
        # is how a calibration chart misleads. Sizing by sqrt(n) keeps every bin visible
        # while showing which ones are thin.
        sizes = 24 + 190 * np.sqrt(ns / ns.max())
        ax.scatter(xs, ys, s=sizes, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted P(lead)")
    ax.set_ylabel("observed rate")
    ax.set_title("Calibration: is P(lead) trustworthy?", loc="left", pad=12)
    # Placed below the diagonal's right end so it never sits on the line it names.
    ax.text(0.98, 0.86, "perfectly calibrated", ha="right", va="top", fontsize=8.5,
            color=INK_MUTED, style="italic", rotation=45, rotation_mode="anchor",
            transform=ax.transAxes)
    ax.text(0.03, 0.97, "marker size = rows in bin", ha="left", va="top", fontsize=8,
            color=INK_MUTED, transform=ax.transAxes)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _finish(ax)
    fig.tight_layout()
    path = out_dir / "calibration.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, rows


def per_brand_performance(y_true, y_pred, brand, out_dir: Path):
    """Accuracy and support per brand -- the dimension the product is organised around.

    A single overall accuracy hides that this model sees ~29k rows for one brand and a
    handful for others. Returns (png, csv, rows) with rows sorted by support.
    """
    df = pd.DataFrame({
        "brand": np.asarray(brand),
        "correct": (np.asarray(y_true).astype(int) == np.asarray(y_pred).astype(int)),
        "actual": np.asarray(y_true).astype(int),
    })
    agg = (
        df.groupby("brand")
        .agg(accuracy=("correct", "mean"), support=("correct", "size"),
             positives=("actual", "sum"))
        .sort_values("support", ascending=False)
    )

    csv_path = out_dir / "per_brand_performance.csv"
    agg.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.42 * len(agg))))
    names = [f"{b}  (n={int(s):,})" for b, s in zip(agg.index, agg["support"])]
    ax.barh(range(len(agg)), agg["accuracy"].to_numpy(), color=BLUE, height=0.62)
    ax.set_yticks(range(len(agg)), names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("accuracy")
    ax.set_title("Lead classifier accuracy by brand (support in labels)", loc="left", pad=12)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, v in enumerate(agg["accuracy"].to_numpy()):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8.5, color=INK_MUTED)
    _finish(ax)
    fig.tight_layout()
    png_path = out_dir / "per_brand_performance.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    rows = [(i + 1, str(b), float(r.accuracy), int(r.support))
            for i, (b, r) in enumerate(agg.iterrows())]
    return png_path, csv_path, rows


def run_summary(train_test_pairs, per_day, out_dir: Path) -> Path:
    """One chart that answers the questions a single-run scalar bar cannot.

    MLflow's chart view renders every scalar metric as its own bar, and for a single run a
    lone bar carries no information -- there is nothing to compare it against. This puts
    the numbers that matter side by side instead:

      left  : train vs test for accuracy and F1, so the generalisation gap is visible
              rather than implied by two separate charts
      right : per-day accuracy across the held-out week, with each point sized by how
              many rows backed it -- which is what stops a one-row day reading as a
              perfect day

    `train_test_pairs` is [(label, train_value, test_value), ...];
    `per_day` is [(day, accuracy, support), ...].
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2), width_ratios=[1, 1.25])

    # --- train vs test -----------------------------------------------------------------
    labels = [p[0] for p in train_test_pairs]
    y = np.arange(len(labels))
    h = 0.34
    ax_l.barh(y + h / 2, [p[1] for p in train_test_pairs], height=h, color=BLUE, label="train")
    ax_l.barh(y - h / 2, [p[2] for p in train_test_pairs], height=h, color=ORANGE, label="test")
    ax_l.set_yticks(y, labels)
    ax_l.set_xlim(0, 1.3)          # headroom so labels never collide with the plot edge
    ax_l.set_ylim(len(labels) - 0.5, -0.75)   # inverted, with room for the legend on top
    ax_l.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_l.set_title("Train vs test", loc="left", pad=22)
    ax_l.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax_l.set_axisbelow(True)
    # Legend above the plot: inside it, it lands on the value labels.
    ax_l.legend(frameon=False, fontsize=9, ncol=2, loc="lower left",
                bbox_to_anchor=(0, 1.0), handlelength=1.2)
    for i, p in enumerate(train_test_pairs):
        ax_l.text(p[1] + 0.015, i + h / 2, f"{p[1]:.2f}", va="center", fontsize=8.5,
                  color=INK_MUTED)
        ax_l.text(p[2] + 0.015, i - h / 2, f"{p[2]:.2f}", va="center", fontsize=8.5,
                  color=INK_MUTED)
        # The gap sits to the RIGHT of both bars, on the surface -- inside the bar it is
        # low-contrast text on a saturated fill.
        gap = p[1] - p[2]
        ax_l.text(1.32, i, f"{gap:+.2f}", va="center", ha="right", fontsize=10,
                  fontweight="bold", color=BRICK if gap > 0.10 else INK_MUTED)
    ax_l.text(1.32, -0.62, "gap", va="center", ha="right", fontsize=8, color=INK_MUTED)
    _finish(ax_l)

    # --- per-day, weighted by support --------------------------------------------------
    if per_day:
        days = [d for d, _, _ in per_day]
        accs = [a for _, a, _ in per_day]
        sup = np.array([s for _, _, s in per_day], dtype=float)
        ax_r.plot(days, accs, color=BLUE, linewidth=2, zorder=2)
        sizes = 24 + 210 * np.sqrt(sup / max(sup.max(), 1))
        ax_r.scatter(days, accs, s=sizes, color=BLUE, edgecolors=SURFACE,
                     linewidths=1.5, zorder=3)
        thin = [(d, a, s) for d, a, s in per_day if s < max(sup.max() * 0.1, 10)]
        for d, a, s in thin:
            ax_r.annotate(f"{int(s)} row{'s' if s != 1 else ''}", (d, a),
                          textcoords="offset points", xytext=(0, 12), ha="center",
                          fontsize=8, color=BRICK)
        ax_r.set_ylim(0, 1.08)
        ax_r.set_xlabel("day of held-out week")
        ax_r.set_ylabel("accuracy")
    ax_r.set_title("Per-day accuracy  (marker size = rows)", loc="left", pad=10)
    ax_r.grid(True, color=GRID, linewidth=0.8)
    ax_r.set_axisbelow(True)
    _finish(ax_r)

    fig.tight_layout()
    path = out_dir / "run_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
