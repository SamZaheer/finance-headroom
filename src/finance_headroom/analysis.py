"""Aggregates results/scored.csv (after judgment rows are graded) into the accuracy-by-bucket
table, the per-model consistency rate, and the study heatmaps (1 iteration and N iterations).
Run after fh-score and grading.
"""
import csv
from collections import defaultdict

from . import models
from .paths import FIGURES as FIG_DIR
from .paths import SCORED


def is_correct(row) -> bool:
    if row["auto_correct"] in ("True", "False"):
        return row["auto_correct"] == "True"
    return row.get("comparability_flag") == "correct"  # manual-graded judgment items


def summarize(rows, ok=is_correct) -> dict:
    """{(model, tool_condition, bucket): accuracy}, averaging over every row given (items x repeats)."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["model"], r["tool_condition"], r["bucket"])].append(ok(r))
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def iteration_heatmaps(rows, prefix, title, ok=is_correct, fig_dir=None):
    """Writes <prefix>_1_iteration.png (repeat 1 only) and, when repeats exist,
    <prefix>_<N>_iterations.png (mean over all N). Shared by fh-analyze and fh-gym."""
    first = [r for r in rows if int(r.get("repeat") or 1) == 1]
    n = max(int(r.get("repeat") or 1) for r in rows)
    fig_dir = fig_dir or FIG_DIR
    outs = [(fig_dir / f"{prefix}_1_iteration.png", first, f"single iteration: repeat 1 only, {len(first)} runs")]
    if n > 1:
        outs.append((fig_dir / f"{prefix}_{n}_iterations.png", rows, f"mean over {n} iterations per question, {len(rows)} runs"))
    for out, rs, subtitle in outs:
        plot_heatmap(summarize(rs, ok), f"{title}\n({subtitle})", out, title_fontsize=12)
    return [o for o, _, _ in outs]


def main():
    rows = list(csv.DictReader(SCORED.open()))
    summary = summarize(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[(r["model"], r["tool_condition"], r["bucket"])] += 1

    print(f"{'model':<8} {'tool':<10} {'bucket':<34} {'n':>3} {'accuracy':>9}")
    for key in sorted(summary):
        print(f"{key[0]:<8} {key[1]:<10} {key[2]:<34} {counts[key]:>3} {summary[key]:>9.0%}")

    report_consistency(rows)

    try:
        for out in iteration_heatmaps(rows, "study_heatmap", "Study accuracy by bucket, model, and tool condition"):
            print(f"wrote {out}")
    except ImportError:
        print("matplotlib not installed -- skipped figure, table above is enough for now")


def report_consistency(rows):
    """With --repeats > 1, accuracy cells are means over repeats. Consistency is the share of
    (model, condition, item) groups whose repeats all got the same outcome -- low consistency
    means a cell's number is sampling noise, not a stable tendency."""
    runs = defaultdict(list)
    for r in rows:
        runs[(r["model"], r["tool_condition"], r["item_id"])].append(is_correct(r))
    repeated = {k: v for k, v in runs.items() if len(v) > 1}
    if not repeated:
        return
    print(f"\n{'model':<8} {'tool':<10} {'items':>5} {'repeats':>8} {'consistent':>11}")
    by_mc = defaultdict(list)
    for (m, c, _), v in repeated.items():
        by_mc[(m, c)].append(v)
    for (m, c), groups in sorted(by_mc.items()):
        same = sum(len(set(g)) == 1 for g in groups)
        print(f"{m:<8} {c:<10} {len(groups):>5} {sum(map(len, groups)) / len(groups):>8.1f} {same / len(groups):>10.0%}")
    flaky = sorted(k for k, v in repeated.items() if len(set(v)) > 1)
    if flaky:
        print("inconsistent across repeats:", ", ".join(f"{m}/{c}/{i}" for m, c, i in flaky))


# Validated palette (see the dataviz skill's references/palette.md).
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
# Sequential blue ramp, lightest -> darkest, used here on (1 - accuracy) so a bigger
# gap (more headroom / more concerning) reads as more ink -- not on accuracy directly,
# which would make the boring 100% cells the darkest ones.
SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281", "#0d366b"]
BUCKET_LABELS = {
    "tech_control": "Tech: control (single-doc)",
    "tech_clean_comparison": "Tech: clean comparison (no accounting break)",
    "tech_gaap_adjustment": "Tech: GAAP/non-GAAP reconciliation",
    "tech_guidance_verification": "Tech: guidance vs. actual (beat/meet/miss)",
    "multi_sector_comparability_break": "Multi-sector: comparability break (split evidence, distractor)",
    "airline_cross_entity": "Airline: cross-entity tracking (5 airlines)",
    "airline_comparability_break": "Airline: comparability break (Hawaiian acquisition)",
    "airline_control": "Airline: control (single clear answer)",
    "airline_tradeoff": "Airline: genuine trade-off",
}
def display_name(model_id: str) -> str:
    """'claude-sonnet-5' -> 'Claude Sonnet 5', 'openai/gpt-5.1' -> 'GPT-5.1'."""
    name = model_id.split("/")[-1]
    if name.lower().startswith("gpt"):
        return name.upper()
    return " ".join(w.capitalize() for w in name.split("-"))


# column headers use the actual model IDs the run was configured with (models.py / .env)
_IDS = {"claude": models.CLAUDE_MODEL, "opus": models.OPUS_MODEL, "gpt": models.GPT_MODEL}
# long names wrap after the first word ("Claude\nSonnet 5") so neighbouring headers don't collide
COLUMN_LABELS = {(k, c): f"{display_name(_IDS[k]).replace(' ', chr(10), 1)}\n{'no tool' if c == 'no_tool' else '+ tool'}"
                 for k in _IDS for c in ("no_tool", "tool")}


def _cell_color(accuracy: float) -> str:
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("gap_seq", SEQUENTIAL_BLUE)
    return cmap(min(max(1 - accuracy, 0), 1))


def plot_heatmap(summary, title, out_path, title_fontsize=13):
    """summary: {(model, tool_condition, bucket): accuracy}. Shared by fh-analyze and fh-replay."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    buckets = sorted({k[2] for k in summary})
    conditions = [("claude", "no_tool"), ("claude", "tool"), ("opus", "no_tool"), ("opus", "tool"),
                  ("gpt", "no_tool"), ("gpt", "tool")]

    # worst (lowest-accuracy, most attention-worthy) bucket at the top
    mean_acc = {b: sum(summary.get((m, t, b), 0) for m, t in conditions) / len(conditions) for b in buckets}
    buckets = sorted(buckets, key=lambda b: mean_acc[b])

    n_rows, n_cols = len(buckets), len(conditions)
    fig, ax = plt.subplots(figsize=(1.6 * n_cols, 0.9 * n_rows + 1.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    cell_w, cell_h = 1.0, 1.0
    for row, bucket in enumerate(buckets):
        # top row first: row 0 drawn at the top, so flip y for the rectangle placement
        y = n_rows - 1 - row
        for col, cond in enumerate(conditions):
            acc = summary.get((cond[0], cond[1], bucket), 0)
            color = _cell_color(acc)
            ax.add_patch(plt.Rectangle((col, y), cell_w - 0.06, cell_h - 0.06, facecolor=color, edgecolor="none"))
            # pick text ink by cell luminance so it always stays legible
            r, g, b = to_rgb(color)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = INK if luminance > 0.55 else "#ffffff"
            ax.text(
                col + (cell_w - 0.06) / 2, y + (cell_h - 0.06) / 2, f"{acc:.0%}",
                ha="center", va="center", fontsize=11, color=text_color, fontweight="medium",
            )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_xticks([c + (cell_w - 0.06) / 2 for c in range(n_cols)])
    ax.set_xticklabels([COLUMN_LABELS[c] for c in conditions], color=INK, fontsize=9.5)
    ax.xaxis.tick_top()
    ax.set_yticks([n_rows - 1 - row + (cell_h - 0.06) / 2 for row in range(n_rows)])
    ax.set_yticklabels([BUCKET_LABELS.get(b, b) for b in buckets], color=INK, fontsize=10)

    ax.set_title(title, color=INK, fontsize=title_fontsize, pad=60, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_aspect("equal")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
