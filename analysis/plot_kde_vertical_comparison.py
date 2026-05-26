from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FOLDER = ROOT / "results_R_30"
OUTPUT_DIR = ROOT / "plots_30" / "kde_vertical_comparison"


def set_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
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


def load_raw_results(results_folder: Path) -> pd.DataFrame:
    frames = []

    for raw_path in sorted(results_folder.glob("*/*/raw_results.csv")):
        learner = raw_path.parent.parent.name
        dimension = raw_path.parent.name
        df = pd.read_csv(raw_path)
        df["learner"] = learner
        df["dimension"] = dimension
        df["dimensionality"] = dimension.replace("d", "")
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No raw_results.csv files found under {results_folder}")

    return pd.concat(frames, ignore_index=True)


def make_vertical_kde_plots(all_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for learner in sorted(all_df["learner"].unique()):
        learner_df = all_df[all_df["learner"] == learner]

        for dim in sorted(learner_df["dimensionality"].unique(), key=int):
            subset = learner_df[learner_df["dimensionality"] == dim]
            if subset.empty:
                continue

            x_min = min(subset["pehe"].min(), subset["pehe_plug"].min())
            x_max = max(subset["pehe"].max(), subset["pehe_plug"].max())
            x_padding = 0.05 * (x_max - x_min) if x_max > x_min else 0.05

            fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            tuners = sorted(subset["tuner"].unique())
            palette = sns.color_palette("colorblind", n_colors=len(tuners))

            for i, tuner in enumerate(tuners):
                pehe_data = subset[subset["tuner"] == tuner]["pehe"]
                pehe_mean = pehe_data.mean()

                sns.kdeplot(
                    pehe_data,
                    ax=axes[0],
                    label=tuner,
                    fill=True,
                    alpha=0.3,
                    color=palette[i],
                )
                axes[0].axvline(
                    pehe_mean,
                    linestyle="--",
                    linewidth=1.5,
                    color=palette[i],
                )

            axes[0].set_title("True PEHE")
            axes[0].set_ylabel("Density")

            for i, tuner in enumerate(tuners):
                plug_data = subset[subset["tuner"] == tuner]["pehe_plug"]
                plug_mean = plug_data.mean()

                sns.kdeplot(
                    plug_data,
                    ax=axes[1],
                    label=tuner,
                    fill=True,
                    alpha=0.3,
                    color=palette[i],
                )
                axes[1].axvline(
                    plug_mean,
                    linestyle="--",
                    linewidth=1.5,
                    color=palette[i],
                )

            axes[1].set_title("Plug-in PEHE")
            axes[1].set_xlabel("Error")
            axes[1].set_ylabel("Density")

            axes[0].set_xlim(x_min - x_padding, x_max + x_padding)

            handles, labels = axes[0].get_legend_handles_labels()
            legend_y = 0.955 if dim in {"1", "2"} else 0.985
            layout_top = 0.92 if dim in {"1", "2"} else 0.95
            fig.legend(
                handles,
                labels,
                title="Tuner",
                loc="upper right",
                ncol=len(tuners),
                bbox_to_anchor=(0.98, legend_y),
            )
            fig.suptitle(f"{learner.upper()} — {dim}D", fontsize=18)
            fig.tight_layout(rect=[0, 0, 1, layout_top])

            filename = output_dir / f"kde_{learner}_{dim}D_vertical.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {filename}")


def main() -> None:
    set_theme()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_df = load_raw_results(RESULTS_FOLDER)
    print(f"Loaded {len(all_df)} rows from raw results.")
    print(f"Writing plots to: {OUTPUT_DIR}")

    make_vertical_kde_plots(all_df, OUTPUT_DIR)
    print("All vertical comparison plots saved.")


if __name__ == "__main__":
    main()
