from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FOLDER = ROOT / "results_R_30"
OUTPUT_BASE = ROOT / "plots_30" / "convergence_plots"
LEARNERS = ["x_cb", "x_rf"]
DIMENSIONS = ["1d", "2d", "4d", "6d"]
TUNERS = ["bayes", "random"]
TUNER_COLORS = {
    "bayes": "#2A6F97",
    "random": "#C8553D",
}


def set_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#C8D1DA",
            "axes.labelcolor": "#243447",
            "axes.titlecolor": "#243447",
            "xtick.color": "#243447",
            "ytick.color": "#243447",
            "grid.color": "#D8DEE4",
            "grid.alpha": 0.6,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
            "legend.title_fontsize": 15,
            "figure.titlesize": 18,
        }
    )


def load_convergence_data(results_folder: Path) -> pd.DataFrame:
    all_frames = []

    for learner in LEARNERS:
        for dimension in DIMENSIONS:
            for tuner in TUNERS:
                pattern = (
                    results_folder
                    / learner
                    / dimension
                    / "convergence"
                    / tuner
                    / "convergence_R*.csv"
                )

                for conv_path in sorted(pattern.parent.glob(pattern.name)):
                    rep = int(conv_path.stem.split("R")[-1])
                    df = pd.read_csv(conv_path)
                    df["learner"] = learner
                    df["dimension"] = dimension
                    df["tuner"] = tuner
                    df["rep"] = rep
                    all_frames.append(df)

    if not all_frames:
        raise FileNotFoundError(f"No convergence CSVs found under {results_folder}")

    return pd.concat(all_frames, ignore_index=True)


def build_summary(convergence_df: pd.DataFrame) -> pd.DataFrame:
    final_scores = (
        convergence_df.sort_values("iteration")
        .groupby(["learner", "dimension", "tuner", "rep"], as_index=False)
        .tail(1)[["learner", "dimension", "tuner", "rep", "best_so_far", "iteration"]]
        .rename(columns={"best_so_far": "final_best_score", "iteration": "final_iteration"})
    )

    return (
        final_scores.groupby(["learner", "dimension", "tuner"], as_index=False)
        .agg(
            mean_final_score=("final_best_score", "mean"),
            std_final_score=("final_best_score", "std"),
            min_final_score=("final_best_score", "min"),
            max_final_score=("final_best_score", "max"),
            n_reps=("rep", "nunique"),
            n_iterations=("final_iteration", lambda s: int(s.iloc[0]) + 1),
        )
        .sort_values(["learner", "dimension", "tuner"])
    )


def aggregate_curves(convergence_df: pd.DataFrame) -> pd.DataFrame:
    averaged = (
        convergence_df.groupby(["learner", "dimension", "tuner", "iteration"], as_index=False)
        .agg(
            mean_best_so_far=("best_so_far", "mean"),
            sd_best_so_far=("best_so_far", "std"),
            n_reps=("rep", "nunique"),
        )
        .sort_values(["learner", "dimension", "tuner", "iteration"])
    )
    averaged["se_best_so_far"] = averaged["sd_best_so_far"] / averaged["n_reps"].pow(0.5)
    return averaged


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(length=0)


def plot_convergence(averaged: pd.DataFrame, output_base: Path) -> None:
    for learner in LEARNERS:
        for dimension in DIMENSIONS:
            subset = averaged[
                (averaged["learner"] == learner) & (averaged["dimension"] == dimension)
            ]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(10.5, 6.2))

            for tuner in TUNERS:
                tuner_data = subset[subset["tuner"] == tuner].sort_values("iteration")
                if tuner_data.empty:
                    continue

                ax.plot(
                    tuner_data["iteration"],
                    tuner_data["mean_best_so_far"],
                    color=TUNER_COLORS[tuner],
                    linewidth=2.8,
                    label=tuner.capitalize(),
                )
                band = tuner_data["se_best_so_far"].fillna(0.0)
                ax.fill_between(
                    tuner_data["iteration"],
                    tuner_data["mean_best_so_far"] - band,
                    tuner_data["mean_best_so_far"] + band,
                    color=TUNER_COLORS[tuner],
                    alpha=0.14,
                    linewidth=0,
                )

                final_x = tuner_data["iteration"].iloc[-1]
                final_y = tuner_data["mean_best_so_far"].iloc[-1]
                ax.scatter(final_x, final_y, color=TUNER_COLORS[tuner], s=42, zorder=5)
                ax.text(
                    final_x + 0.6,
                    final_y,
                    f"{final_y:.3f}",
                    color=TUNER_COLORS[tuner],
                    fontsize=12,
                    va="center",
                )

            ax.set_title(
                f"{learner.upper()} {dimension.upper()}  Mean convergence across 30 reps",
                fontsize=17,
                fontweight="bold",
            )
            ax.set_xlabel("Tuning iteration")
            ax.set_ylabel("Best-so-far MSE")
            ax.legend(loc="upper right", frameon=False)
            style_axis(ax)

            output_dir = output_base / learner
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / f"convergence_{dimension}.png", dpi=320, bbox_inches="tight")
            plt.close()


def main() -> None:
    set_theme()
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing: {RESULTS_FOLDER}")
    print(f"Output directory: {OUTPUT_BASE}")

    convergence_df = load_convergence_data(RESULTS_FOLDER)
    print(f"\nLoaded {len(convergence_df)} total convergence records")

    summary = build_summary(convergence_df)
    summary_path = OUTPUT_BASE / "convergence_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    averaged = aggregate_curves(convergence_df)
    plot_convergence(averaged, OUTPUT_BASE)


if __name__ == "__main__":
    main()
