"""Eight reproducible, static research figures; matplotlib only, never seaborn."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.metrics.core import Row

LABELS = {
    "baseline": "Single judge",
    "baseline_no_feedback": "No-feedback control",
    "critique": "Judge + critique",
    "adversarial": "Adversarial critic",
    "ensemble": "Ensemble",
    "execution": "Execution evidence",
    "execution_evidence": "Execution evidence",
}
COLORS = {
    "baseline": "#245E95",
    "baseline_no_feedback": "#606A73",
    "critique": "#AF8500",
    "adversarial": "#C36828",
    "ensemble": "#67833E",
    "execution": "#B4587B",
    "execution_evidence": "#B4587B",
}
MARKERS = {
    "baseline": "o",
    "baseline_no_feedback": "x",
    "critique": "s",
    "adversarial": "^",
    "ensemble": "D",
    "execution": "P",
    "execution_evidence": "P",
}
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2)), (0, (2, 2, 5, 2))]


def generate_figures(rows: list[Row], analysis: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "MPLCONFIGDIR", str((output_dir.parent / ".cache" / "matplotlib").resolve())
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

    provenance = analysis["provenance"]
    final_round = provenance["final_round"]
    conditions = sorted(
        provenance["conditions"], key=lambda c: (list(LABELS).index(c) if c in LABELS else 99, c)
    )
    final = [row for row in rows if row["round"] == final_round]
    prefix = "MOCK VALIDATION · " if provenance["run_kind"] == "mock" else ""
    caution = (
        "Synthetic plumbing check; no empirical model findings. "
        if provenance["run_kind"] == "mock"
        else ""
    )
    note = f"{caution}Final round {final_round}; {provenance['n_tasks']} task clusters. Intervals: 95% task bootstrap."
    palette = {condition: COLORS.get(condition, "#51565B") for condition in conditions}
    paths: list[Path] = []

    def finish(fig: Any, name: str, title: str, figure_note: str = note, top: float = 0.91) -> None:
        fig.suptitle(prefix + title, x=0.5, y=0.99, fontsize=13, fontweight="bold")
        fig.text(0.02, 0.015, figure_note, ha="left", va="bottom", fontsize=8, wrap=True)
        fig.tight_layout(rect=(0, 0.075, 1, top))
        for suffix in ("png", "pdf"):
            path = output_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=300, facecolor="white", metadata={"Title": prefix + title})
            paths.append(path)
        plt.close(fig)

    def score_axis(ax: Any, which: str = "y") -> None:
        axis = ax.yaxis if which == "y" else ax.xaxis
        axis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        (ax.set_ylim if which == "y" else ax.set_xlim)(-0.03, 1.03)

    def gap_axis(ax: Any, which: str = "y") -> None:
        axis = ax.yaxis if which == "y" else ax.xaxis
        axis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:.0f}"))

    def legend_handles() -> list[Any]:
        return [
            Line2D(
                [0],
                [0],
                label=LABELS.get(condition, condition),
                color=palette[condition],
                marker=MARKERS.get(condition, "o"),
                linestyle=LINESTYLES[i % len(LINESTYLES)],
            )
            for i, condition in enumerate(conditions)
        ]

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#62676D",
            "axes.linewidth": 0.8,
            "grid.color": "#E1E4E8",
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
        }
    ):
        # 1. Points are final task × condition observations; no artificial jitter.
        fig, ax = plt.subplots(figsize=(7.1, 5.2))
        ax.plot([0, 1], [0, 1], linestyle="--", color="#454A50", linewidth=1.0, label="J = G")
        for condition in conditions:
            group = [r for r in final if r["condition"] == condition]
            marker = MARKERS.get(condition, "o")
            marker_style = (
                {"edgecolors": "white", "linewidths": 0.35}
                if marker != "x"
                else {"linewidths": 1.0}
            )
            ax.scatter(
                [r["ground_truth_score"] for r in group],
                [r["judge_score"] for r in group],
                marker=marker,
                s=36,
                alpha=0.7,
                color=palette[condition],
                label=LABELS.get(condition, condition),
                **marker_style,
            )
        ax.set(
            xlabel="Hidden test fraction passed G",
            ylabel="Predicted full-correctness probability J",
        )
        score_axis(ax)
        score_axis(ax, "x")
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
        finish(
            fig,
            "judge_vs_ground_truth",
            "Judge score versus hidden-test fraction",
            note + " Coincident points can overlap.",
        )

        # 2. Signed gap: zero line and point intervals make negative values explicit.
        summary = {r["condition"]: r for r in analysis["summary"]}
        fig, ax = plt.subplots(figsize=(7.1, 4.5))
        for index, condition in enumerate(conditions):
            entry = summary[condition]
            estimate, low, high = (
                entry["mean_gap"],
                entry["mean_gap_ci_low"],
                entry["mean_gap_ci_high"],
            )
            if low is not None:
                ax.hlines(index, low, high, color=palette[condition], linewidth=1.6)
                ax.plot([low, high], [index, index], "|", color=palette[condition], markersize=8)
            ax.scatter(
                [estimate],
                [index],
                color=palette[condition],
                marker=MARKERS.get(condition, "o"),
                s=48,
                zorder=3,
            )
        ax.axvline(0, color="#454A50", linewidth=1)
        ax.set_yticks(range(len(conditions)), [LABELS.get(c, c) for c in conditions])
        ax.invert_yaxis()
        ax.set_xlabel("Mean evaluation gap J − G (percentage points)")
        ax.grid(axis="x")
        gap_axis(ax, "x")
        finish(fig, "mean_gap_by_condition", "Evaluation gap by evaluator condition")

        # 3. A pair of comparable panels avoids implying that J and G are calibrated alike.
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), sharex=True, sharey=True)
        for ax, metric, title in zip(
            axes,
            ("mean_judge_score", "mean_ground_truth"),
            ("Judge score J", "Hidden-test fraction G"),
            strict=True,
        ):
            for index, condition in enumerate(conditions):
                points = sorted(
                    (r for r in analysis["rounds"] if r["condition"] == condition),
                    key=lambda r: r["round"],
                )
                x, y = [r["round"] for r in points], [r[metric] for r in points]
                ax.plot(
                    x,
                    y,
                    color=palette[condition],
                    marker=MARKERS.get(condition, "o"),
                    linestyle=LINESTYLES[index % len(LINESTYLES)],
                    markersize=4,
                )
                if all(r[f"{metric}_ci_low"] is not None for r in points):
                    ax.fill_between(
                        x,
                        [r[f"{metric}_ci_low"] for r in points],
                        [r[f"{metric}_ci_high"] for r in points],
                        color=palette[condition],
                        alpha=0.08,
                        linewidth=0,
                    )
            ax.set(title=title, xlabel="Proposer revision round")
            ax.set_xticks(list(range(final_round + 1)))
            score_axis(ax)
            ax.grid(axis="y")
        axes[0].set_ylabel("Mean score")
        fig.legend(
            handles=legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=min(3, len(conditions)),
            frameon=False,
            fontsize=8,
        )
        finish(
            fig,
            "scores_across_rounds",
            "Judge score and objective score across revisions",
            f"{caution}{provenance['n_tasks']} task clusters; round 0 is shared initial code. Shading: 95% task-bootstrap CI.",
            top=0.79,
        )

        # 4. Paired per-task gaps, summarized at every revision round.
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        for index, condition in enumerate(conditions):
            points = sorted(
                (r for r in analysis["rounds"] if r["condition"] == condition),
                key=lambda r: r["round"],
            )
            x = [r["round"] for r in points]
            ax.plot(
                x,
                [r["mean_gap"] for r in points],
                color=palette[condition],
                marker=MARKERS.get(condition, "o"),
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                label=LABELS.get(condition, condition),
            )
            if all(r["mean_gap_ci_low"] is not None for r in points):
                ax.fill_between(
                    x,
                    [r["mean_gap_ci_low"] for r in points],
                    [r["mean_gap_ci_high"] for r in points],
                    color=palette[condition],
                    alpha=0.08,
                    linewidth=0,
                )
        ax.axhline(0, color="#454A50", linewidth=1)
        ax.set(xlabel="Proposer revision round", ylabel="Mean gap J − G (percentage points)")
        ax.set_xticks(list(range(final_round + 1)))
        gap_axis(ax)
        ax.grid(axis="y")
        ax.legend(fontsize=8, frameon=False, ncol=2)
        finish(
            fig,
            "gap_across_rounds",
            "Evaluation gap across proposer revisions",
            f"{caution}{provenance['n_tasks']} task clusters. Shading: 95% task-bootstrap CI.",
        )

        # 5. Calibration target is the all-tests-pass event, never fractional G.
        fig, ax = plt.subplots(figsize=(7.1, 5.2))
        ax.plot([0, 1], [0, 1], "--", color="#454A50", linewidth=1)
        for index, condition in enumerate(conditions):
            bins = [
                r for r in analysis["reliability"] if r["condition"] == condition and r["count"]
            ]
            ax.plot(
                [b["mean_score"] for b in bins],
                [b["empirical_accuracy"] for b in bins],
                color=palette[condition],
                marker=MARKERS.get(condition, "o"),
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                markersize=5,
                label=LABELS.get(condition, condition),
            )
        score_axis(ax)
        score_axis(ax, "x")
        ax.set(
            xlabel="Mean predicted P(all hidden tests pass)",
            ylabel="Observed fraction passing all hidden tests",
        )
        ax.grid(True)
        ax.legend(loc="lower right", frameon=False, fontsize=8)
        finish(
            fig,
            "reliability_curve",
            "Reliability for the full-correctness event",
            f"{caution}Final round {final_round}; 10 equal-width bins. Empty bins omitted; bin counts are in reliability.csv.",
        )

        # 6. Conditional error rate, with undefined denominators clearly distinguished.
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for index, condition in enumerate(conditions):
            entry = summary[condition]
            estimate = entry["false_approval_rate"]
            if estimate is None:
                ax.text(index, 0.04, "Undefined", ha="center", fontsize=8, rotation=90)
                continue
            ax.bar(
                index,
                estimate,
                width=0.58,
                color=palette[condition],
                edgecolor="#454A50",
                linewidth=0.6,
            )
            low, high = entry["false_approval_rate_ci_low"], entry["false_approval_rate_ci_high"]
            if low is not None:
                ax.vlines(index, low, high, color="#30353A", linewidth=1.2)
                ax.plot([index, index], [low, high], "_", color="#30353A", markersize=7)
            ax.text(index, min(estimate + 0.025, 1.04), f"{estimate:.0%}", ha="center", fontsize=9)
        ax.set_xticks(
            range(len(conditions)),
            [
                LABELS.get(c, c)
                .replace(" + ", " +\n")
                .replace("Adversarial critic", "Adversarial\ncritic")
                .replace("Execution evidence", "Execution\nevidence")
                .replace("No-feedback control", "No-feedback\ncontrol")
                for c in conditions
            ],
        )
        ax.set_ylim(0, 1.12)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_ylabel("P(approve | at least one hidden test fails)")
        ax.grid(axis="y")
        finish(fig, "false_approval_by_condition", "False approval rate by evaluator condition")

        # 7. Identical bins and count axes preserve a truthful comparison of spread.
        fig, ax = plt.subplots(figsize=(7.6, 4.8))
        bins = [-1 + index * 0.1 for index in range(21)]
        for index, condition in enumerate(conditions):
            values = [
                r["judge_score"] - r["ground_truth_score"]
                for r in final
                if r["condition"] == condition
            ]
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                color=palette[condition],
                linewidth=1.6,
                linestyle=LINESTYLES[index % len(LINESTYLES)],
                label=LABELS.get(condition, condition),
            )
        ax.axvline(0, color="#454A50", linewidth=1)
        ax.set(
            xlabel="Evaluation gap J − G (percentage points)",
            ylabel="Number of task × replicate observations",
            xlim=(-1, 1),
        )
        gap_axis(ax, "x")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y")
        ax.legend(fontsize=8, frameon=False)
        finish(fig, "gap_distribution", "Distribution of final-round evaluation gaps")

        # 8. Deliberately selected, clearly nonrepresentative failure trajectories.
        ordered = sorted(
            final,
            key=lambda r: (
                -(r["judge_score"] - r["ground_truth_score"]),
                str(r["task_id"]),
                str(r["condition"]),
            ),
        )
        high_gap = [
            r for r in ordered if r["judge_score"] >= 0.8 and r["ground_truth_score"] <= 0.5
        ]
        selected, seen = [], set()
        for row in high_gap + ordered:
            if row["task_id"] not in seen:
                selected.append(row)
                seen.add(row["task_id"])
            if len(selected) == 4:
                break
        ncols = 2 if len(selected) > 1 else 1
        nrows = (len(selected) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(9.0 if ncols == 2 else 6.3, 3.15 * nrows + 1.1),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for ax, chosen in zip(axes.flat, selected, strict=False):
            trajectory = sorted(
                (
                    r
                    for r in rows
                    if r["task_id"] == chosen["task_id"]
                    and r["condition"] == chosen["condition"]
                    and str(r.get("replicate", 0)) == str(chosen.get("replicate", 0))
                ),
                key=lambda r: r["round"],
            )
            x = [r["round"] for r in trajectory]
            ax.plot(
                x,
                [r["judge_score"] for r in trajectory],
                color="#245E95",
                marker="o",
                label="Judge J",
            )
            ax.plot(
                x,
                [r["ground_truth_score"] for r in trajectory],
                color="#AF8500",
                marker="s",
                linestyle="--",
                label="Hidden fraction G",
            )
            ax.set(
                title=f"{chosen['task_id']} · {LABELS.get(chosen['condition'], chosen['condition'])}",
                xlabel="Proposer revision round",
                ylabel="Score",
            )
            ax.set_xticks(list(range(final_round + 1)))
            score_axis(ax)
            ax.grid(axis="y")
        for ax in list(axes.flat)[len(selected) :]:
            ax.set_visible(False)
        fig.legend(
            handles=[
                Line2D([0], [0], color="#245E95", marker="o", label="Judge J"),
                Line2D(
                    [0], [0], color="#AF8500", marker="s", linestyle="--", label="Hidden fraction G"
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=2,
            frameon=False,
            fontsize=9,
        )
        finish(
            fig,
            "task_trajectory_examples",
            "Selected per-task trajectories",
            f"{caution}Selected by final gap, one condition per task; examples are not representative estimates.",
            top=0.86,
        )
    return paths
