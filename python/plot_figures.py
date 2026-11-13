from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from python.models import MODEL_NAMES

COLORS = ("#2463A5", "#159A8C", "#D98232", "#8362A6", "#C34B60", "#607480")


def load_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_figure(figure: Figure, directory: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def style_auc(axis: Axes, title: str) -> None:
    axis.set_title(title, loc="left", fontweight="bold", pad=12)
    axis.set_ylabel("Test AUC")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def metric_values(rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    values = [r["metrics"][metric] for r in rows]
    return (np.array([np.nan if v["mean"] is None else v["mean"] for v in values]),
            np.array([0 if v["std"] is None else v["std"] for v in values]))


def main_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {r["model"]: r for r in summary if r["protocol"] == "window_aligned" and r["ablation"] == "none"}
    return [lookup[name] for name in MODEL_NAMES if name in lookup]


def plot_ladder(summary: list[dict[str, Any]], directory: Path) -> None:
    rows = main_rows(summary)
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(11, 4.6), layout="constrained")
    x = np.arange(len(rows))
    for index, (metric, label) in enumerate((("overall_auc", "Overall"), ("step_1_5_auc", "First 5 answers"), ("step_1_10_auc", "First 10 answers"))):
        values, errors = metric_values(rows, metric)
        axis.bar(x + (index - 1) * 0.25, values, width=0.24, yerr=errors,
                 label=label, color=COLORS[index], capsize=3)
    axis.set_xticks(x, [r["model"] for r in rows])
    style_auc(axis, "Controlled model comparison")
    axis.legend(loc="upper right", frameon=False)
    fig.supxlabel("Error bars: sample SD across training seeds; undefined AUC omitted", fontsize=9)
    save_figure(fig, directory, "fig1_controlled_ladder")


def plot_history_rarity(summary: list[dict[str, Any]], directory: Path) -> None:
    rows = [r for r in main_rows(summary) if r["model"] in {"CausalSeqOnly", "HSF_Concat", "HSF_Gate", "SAKT"}]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    history = ["step_1_5_auc", "step_6_10_auc", "step_11_30_auc", "step_gt_30_auc"]
    rarity = ["freq_0_unseen_auc", "freq_1_5_rare_auc", "freq_6_20_med_auc", "freq_gt_20_common_auc"]
    for index, row in enumerate(rows):
        values = [metric_values([row], key)[0][0] for key in history]
        errors = [metric_values([row], key)[1][0] for key in history]
        axes[0].errorbar(range(4), values, yerr=errors, marker="o", color=COLORS[index], label=row["model"], capsize=3)
        values = [metric_values([row], key)[0][0] for key in rarity]
        errors = [metric_values([row], key)[1][0] for key in rarity]
        axes[1].bar(np.arange(4) + (index - (len(rows) - 1) / 2) * 0.2, values,
                    yerr=errors, width=0.19, color=COLORS[index], label=row["model"], capsize=2)
    axes[0].set_xticks(range(4), ["1–5", "6–10", "11–30", ">30"])
    axes[1].set_xticks(range(4), ["0 (unseen)", "1–5", "6–20", ">20"])
    axes[0].set_xlabel("Original student answer number")
    axes[1].set_xlabel("Item occurrences in full training set")
    style_auc(axes[0], "Short-history groups")
    style_auc(axes[1], "Item-frequency groups")
    axes[0].legend(loc="lower right", frameon=False)
    fig.supxlabel("Missing bars/points indicate empty or single-class groups (see counts in results.json)", fontsize=9)
    save_figure(fig, directory, "fig2_short_history_and_rarity")


def plot_ablation(summary: list[dict[str, Any]], directory: Path) -> None:
    lookup = {r["ablation"]: r for r in summary if r["model"] == "HSF_Gate" and r["protocol"] == "window_aligned"}
    order = [key for key in ("none", "item_prior", "user_cum", "recent_state") if key in lookup]
    if len(order) < 2:
        return
    rows = [lookup[key] for key in order]
    fig, axis = plt.subplots(figsize=(9, 4.6), layout="constrained")
    for index, (metric, label) in enumerate((("overall_auc", "Overall"), ("step_1_5_auc", "First 5 answers"))):
        values, errors = metric_values(rows, metric)
        axis.bar(np.arange(len(rows)) + (index - 0.5) * 0.32, values, width=0.3,
                 yerr=errors, label=label, color=COLORS[index], capsize=3)
    labels = {"none": "Full HSF Gate", "item_prior": "Without priors + bias", "user_cum": "Without cumulative", "recent_state": "Without recent state"}
    axis.set_xticks(range(len(rows)), [labels[key] for key in order])
    style_auc(axis, "Statistical feature ablations")
    axis.legend(loc="upper right", frameon=False)
    save_figure(fig, directory, "fig3_feature_ablation")


def gate_frame(results: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for result in results:
        if (result["model"], result["protocol"], result["ablation"]) != ("HSF_Gate", "window_aligned", "none"):
            continue
        with np.load(result["predictions"], allow_pickle=False) as archive:
            keep = (archive["original_step"] < 50) & np.isfinite(archive["stat_weight"])
            frames.append(pd.DataFrame({"user": archive["user"][keep],
                                        "step": archive["original_step"][keep] + 1,
                                        "weight": archive["stat_weight"][keep]}))
    if not frames:
        return pd.DataFrame(columns=["user", "step", "weight"])
    # Average seeds for each student/step before computing between-student SD.
    return pd.concat(frames).groupby(["user", "step"], as_index=False)["weight"].mean()


def plot_gate(results: list[dict[str, Any]], directory: Path) -> None:
    frame = gate_frame(results)
    if frame.empty:
        return
    aggregate = frame.groupby("step")["weight"].agg(["mean", "std", "count"]).fillna(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    x, mean, sd = aggregate.index.to_numpy(), aggregate["mean"].to_numpy(), aggregate["std"].to_numpy()
    axes[0].plot(x, mean, color=COLORS[0], label="Population mean")
    axes[0].fill_between(x, mean - sd, mean + sd, color=COLORS[0], alpha=0.2, label="± 1 student SD")
    users = frame.groupby("user").size().sort_index()
    longest = users[users == users.max()].index[:3]
    for index, user in enumerate(longest):
        case = frame[frame["user"] == user].sort_values("step")
        axes[1].plot(case["step"], case["weight"], color=COLORS[index], label=f"Student {user}")
    for axis, title in zip(axes, ("Population gate dynamics", "Deterministically selected longest cases")):
        axis.set_title(title, loc="left", fontweight="bold", pad=12)
        axis.set_xlabel("Original student answer number")
        axis.set_ylabel("Statistical branch weight (1 − mean gate)")
        axis.set_ylim(0, 1)
        axis.set_xlim(1, 50)
        axis.legend(frameon=False, loc="lower right")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Seeds averaged per student; available students vary by step. No trend is imposed.", fontsize=9)
    save_figure(fig, directory, "fig4_gate_dynamics")


def plot_speed(benchmarks: list[dict[str, Any]], directory: Path) -> None:
    if not benchmarks:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
    settings = sorted({(r["backend"], r["threads"]) for r in benchmarks}, reverse=True)
    for index, (backend, threads) in enumerate(settings):
        rows = sorted((r for r in benchmarks if (r["backend"], r["threads"]) == (backend, threads)), key=lambda r: r["interactions"])
        x = [r["interactions"] for r in rows]
        label = "Python" if backend == "python" else f"C++ / {threads} threads"
        axes[0].plot(x, [r["median_ms"] for r in rows], marker="o", label=label, color=COLORS[index])
        axes[1].plot(x, [r["speedup"] for r in rows], marker="o", label=label, color=COLORS[index])
    for axis, title in zip(axes, ("OOF fit + feature extraction", "Speedup against Python reference")):
        axis.set_title(title, loc="left", fontweight="bold", pad=12)
        axis.set_xlabel("Interaction count (log scale)")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median elapsed time (ms, log scale)")
    axes[1].set_ylabel("Python time / backend time")
    axes[0].legend(frameon=False, fontsize=8)
    has_cpp = any(row["backend"] == "cpp" for row in benchmarks)
    note = "measured parity < 1e−6" if has_cpp else "C++ unavailable; no cross-language comparison"
    fig.supxlabel("Prepacked input; allocation + thread startup included; " + note, fontsize=9)
    save_figure(fig, directory, "fig5_cpp_oof_speedup")


def plot_all(output: Path) -> None:
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    summary = load_json(output / "summary.json")
    plot_ladder(summary, directory)
    plot_history_rarity(summary, directory)
    plot_ablation(summary, directory)
    plot_gate(load_json(output / "results.json"), directory)
    plot_speed(load_json(output / "benchmarks.json"), directory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render cached HSF-KT results; no Test inference")
    parser.add_argument("output", type=Path)
    plot_all(parser.parse_args().output.resolve())


if __name__ == "__main__":
    main()
