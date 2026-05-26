from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FOLDER = ROOT / "results_R_30"
OUTPUT_BASE = ROOT / "plots_30" / "surrogate_plots"
LEARNERS = ["x_cb", "x_rf"]
DIMENSIONS = ["1d", "2d", "4d", "6d"]
COLORS = {
    "obs": "#C8553D",
    "pred": "#2A6F97",
    "unc": "#7A8C99",
    "err": "#6A4C93",
    "accent": "#7FB069",
    "grid": "#D8DEE4",
    "text": "#243447",
}


def clean_params_string(value: str) -> str:
    cleaned = re.sub(r"np\.[A-Za-z0-9_]+\(([^()]+)\)", r"\1", value)
    cleaned = cleaned.replace("np.True_", "True").replace("np.False_", "False")
    return cleaned


def parse_params(value: str) -> dict[str, float]:
    parsed = ast.literal_eval(clean_params_string(value))
    return {str(key): float(val) for key, val in parsed.items()}


def load_surrogate_data(results_folder: Path) -> pd.DataFrame:
    all_frames = []

    for learner in LEARNERS:
        for dimension in DIMENSIONS:
            surrogate_dir = results_folder / learner / dimension / "convergence" / "bayes"
            for path in sorted(surrogate_dir.glob("surrogate_R*.csv")):
                rep = int(path.stem.split("R")[-1])
                df = pd.read_csv(path)
                if df.empty:
                    continue

                df["learner"] = learner
                df["dimension"] = dimension
                df["rep"] = rep
                df["abs_error"] = (df["score"] - df["pred_mu"]).abs()
                df["params_dict"] = df["params"].map(parse_params)

                params_df = pd.DataFrame(df["params_dict"].tolist())
                params_df.columns = [f"param::{col}" for col in params_df.columns]
                df = pd.concat([df, params_df], axis=1)
                all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError(f"No surrogate CSVs found under {results_folder}")

    return pd.concat(all_frames, ignore_index=True)


def set_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C8D1DA",
            "axes.labelcolor": COLORS["text"],
            "axes.titlecolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.6,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 13,
            "figure.titlesize": 18,
        }
    )


def aggregate_iteration_metrics(df: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        df.groupby(["learner", "dimension", "iter"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_pred_mu=("pred_mu", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_abs_error=("abs_error", "mean"),
            se_score=("score", lambda s: s.std(ddof=1) / math.sqrt(s.nunique() if s.nunique() else 1)),
            se_pred_mu=("pred_mu", lambda s: s.std(ddof=1) / math.sqrt(s.nunique() if s.nunique() else 1)),
            se_pred_std=("pred_std", lambda s: s.std(ddof=1) / math.sqrt(s.nunique() if s.nunique() else 1)),
            se_abs_error=("abs_error", lambda s: s.std(ddof=1) / math.sqrt(s.nunique() if s.nunique() else 1)),
            n_reps=("rep", "nunique"),
        )
        .sort_values(["learner", "dimension", "iter"])
    )
    return aggregated


def add_ribbon(ax: plt.Axes, x: pd.Series, mean: pd.Series, se: pd.Series, color: str) -> None:
    band = se.fillna(0.0)
    ax.fill_between(x, mean - band, mean + band, color=color, alpha=0.14, linewidth=0)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(length=0)


def plot_diagnostics(aggregated: pd.DataFrame) -> None:
    for learner in LEARNERS:
        for dimension in DIMENSIONS:
            subset = aggregated[
                (aggregated["learner"] == learner) & (aggregated["dimension"] == dimension)
            ]
            if subset.empty:
                continue

            fig, axes = plt.subplots(
                3, 1, figsize=(10.5, 9), sharex=True, gridspec_kw={"hspace": 0.18}
            )

            axes[0].plot(subset["iter"], subset["mean_score"], color=COLORS["obs"], linewidth=2.6)
            add_ribbon(axes[0], subset["iter"], subset["mean_score"], subset["se_score"], COLORS["obs"])
            axes[0].plot(subset["iter"], subset["mean_pred_mu"], color=COLORS["pred"], linewidth=2.6)
            add_ribbon(axes[0], subset["iter"], subset["mean_pred_mu"], subset["se_pred_mu"], COLORS["pred"])
            axes[0].set_ylabel("MSE")
            axes[0].set_title("Observed score and surrogate mean")
            style_axis(axes[0])

            axes[1].plot(subset["iter"], subset["mean_pred_std"], color=COLORS["unc"], linewidth=2.4)
            add_ribbon(axes[1], subset["iter"], subset["mean_pred_std"], subset["se_pred_std"], COLORS["unc"])
            axes[1].set_ylabel("Pred. std")
            axes[1].set_title("Model uncertainty")
            style_axis(axes[1])

            axes[2].plot(subset["iter"], subset["mean_abs_error"], color=COLORS["err"], linewidth=2.4)
            add_ribbon(axes[2], subset["iter"], subset["mean_abs_error"], subset["se_abs_error"], COLORS["err"])
            axes[2].set_ylabel("|score - pred|")
            axes[2].set_xlabel("Bayes iteration")
            axes[2].set_title("Prediction error")
            style_axis(axes[2])

            legend_handles = [
                Line2D([0], [0], color=COLORS["obs"], lw=2.8, label="Observed score"),
                Line2D([0], [0], color=COLORS["pred"], lw=2.8, label="Surrogate mean"),
            ]
            fig.legend(handles=legend_handles, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
            fig.suptitle(
                f"{learner.upper()} {dimension.upper()}  Bayes surrogate diagnostics",
                y=0.995,
                fontsize=17,
                fontweight="bold",
            )

            output_dir = OUTPUT_BASE / learner
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"diagnostics_{dimension}.png", dpi=320, bbox_inches="tight")
            plt.close()


def make_bins(series: pd.Series, max_bins: int = 18) -> pd.Series:
    n_bins = min(max_bins, max(6, series.nunique()))
    return pd.qcut(series.rank(method="first"), q=n_bins, duplicates="drop")


def summarize_binned(df: pd.DataFrame, param_col: str) -> pd.DataFrame:
    bins = make_bins(df[param_col])
    binned = (
        df.assign(_bin=bins)
        .groupby("_bin", observed=True, as_index=False)
        .agg(
            param_value=(param_col, "mean"),
            score_mean=("score", "mean"),
            pred_mean=("pred_mu", "mean"),
            pred_std_mean=("pred_std", "mean"),
            abs_error_mean=("abs_error", "mean"),
            n=("score", "size"),
        )
        .sort_values("param_value")
    )
    return binned


def plot_1d_fit(df: pd.DataFrame, learner: str, dimension: str, param_col: str) -> None:
    plot_df = df[(df["learner"] == learner) & (df["dimension"] == dimension)].copy()
    binned = summarize_binned(plot_df, param_col)
    param_label = param_col.replace("param::", "")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.8), sharex=True, gridspec_kw={"hspace": 0.14})

    axes[0].scatter(plot_df[param_col], plot_df["score"], s=18, alpha=0.12, color=COLORS["obs"], edgecolors="none")
    axes[0].plot(binned["param_value"], binned["score_mean"], color=COLORS["obs"], linewidth=2.6)
    axes[0].plot(binned["param_value"], binned["pred_mean"], color=COLORS["pred"], linewidth=2.6)
    axes[0].fill_between(
        binned["param_value"],
        binned["pred_mean"] - 2 * binned["pred_std_mean"].fillna(0.0),
        binned["pred_mean"] + 2 * binned["pred_std_mean"].fillna(0.0),
        color=COLORS["pred"],
        alpha=0.14,
        linewidth=0,
    )
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Binned pooled fit across 30 reps")
    style_axis(axes[0])

    axes[1].plot(binned["param_value"], binned["abs_error_mean"], color=COLORS["err"], linewidth=2.4)
    axes[1].fill_between(
        binned["param_value"],
        0,
        binned["abs_error_mean"],
        color=COLORS["err"],
        alpha=0.12,
        linewidth=0,
    )
    axes[1].set_ylabel("Mean abs. error")
    axes[1].set_xlabel(param_label)
    axes[1].set_title("Calibration by parameter value")
    style_axis(axes[1])

    legend_handles = [
        Line2D([0], [0], color=COLORS["obs"], lw=2.8, label="Observed score"),
        Line2D([0], [0], color=COLORS["pred"], lw=2.8, label="Surrogate mean"),
        Line2D([0], [0], color=COLORS["err"], lw=2.8, label="Mean abs. error"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncols=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(
        f"{learner.upper()} {dimension.upper()}  Surrogate fit",
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )

    output_dir = OUTPUT_BASE / learner
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"surrogate_fit_{dimension}.png", dpi=320, bbox_inches="tight")
    plt.close()


def plot_parameter_slices(df: pd.DataFrame, learner: str, dimension: str, param_cols: list[str]) -> None:
    subset = df[(df["learner"] == learner) & (df["dimension"] == dimension)].copy()
    n_cols = 2
    n_rows = math.ceil(len(param_cols) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4.3 * n_rows))
    axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, param_col in zip(axes.flat, param_cols):
        ax.set_visible(True)
        binned = summarize_binned(subset, param_col)
        label = param_col.replace("param::", "")

        ax.plot(binned["param_value"], binned["score_mean"], color=COLORS["obs"], linewidth=2.2)
        ax.plot(binned["param_value"], binned["pred_mean"], color=COLORS["pred"], linewidth=2.2)
        ax.fill_between(
            binned["param_value"],
            binned["pred_mean"] - binned["pred_std_mean"].fillna(0.0),
            binned["pred_mean"] + binned["pred_std_mean"].fillna(0.0),
            color=COLORS["pred"],
            alpha=0.12,
            linewidth=0,
        )
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("MSE")
        style_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=COLORS["obs"], lw=2.6, label="Observed score"),
        Line2D([0], [0], color=COLORS["pred"], lw=2.6, label="Surrogate mean"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(
        f"{learner.upper()} {dimension.upper()}  Parameter slices",
        y=0.995,
        fontsize=17,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir = OUTPUT_BASE / learner
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"parameter_slices_{dimension}.png", dpi=320, bbox_inches="tight")
    plt.close()


def plot_parameter_views(df: pd.DataFrame) -> None:
    param_cols = [col for col in df.columns if col.startswith("param::")]

    for learner in LEARNERS:
        for dimension in DIMENSIONS:
            subset = df[(df["learner"] == learner) & (df["dimension"] == dimension)]
            current_param_cols = [col for col in param_cols if not subset.empty and subset[col].notna().any()]
            if not current_param_cols:
                continue

            if len(current_param_cols) == 1:
                plot_1d_fit(df, learner, dimension, current_param_cols[0])
            else:
                plot_parameter_slices(df, learner, dimension, current_param_cols)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    final_per_rep = (
        df.sort_values(["learner", "dimension", "rep", "iter"])
        .groupby(["learner", "dimension", "rep"], as_index=False)
        .tail(1)
    )
    return (
        final_per_rep.groupby(["learner", "dimension"], as_index=False)
        .agg(
            n_reps=("rep", "nunique"),
            n_iterations=("iter", lambda s: int(s.iloc[0]) + 1),
            mean_final_score=("score", "mean"),
            mean_pred_std=("pred_std", "mean"),
            mean_abs_error=("abs_error", "mean"),
        )
        .sort_values(["learner", "dimension"])
    )


def main() -> None:
    set_theme()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing surrogate files in: {RESULTS_FOLDER}")
    print(f"Output directory: {OUTPUT_BASE}")

    surrogate_df = load_surrogate_data(RESULTS_FOLDER)
    print(f"\nLoaded {len(surrogate_df)} surrogate rows")

    summary = build_summary(surrogate_df)
    summary_path = OUTPUT_BASE / "surrogate_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    aggregated = aggregate_iteration_metrics(surrogate_df)
    plot_diagnostics(aggregated)
    plot_parameter_views(surrogate_df)


if __name__ == "__main__":
    main()
