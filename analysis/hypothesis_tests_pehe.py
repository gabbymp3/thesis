from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp, ttest_rel


ROOT = Path(__file__).resolve().parents[1]
RESULTS_FOLDER = ROOT / "results_R_30"
OUTPUT_PATH = ROOT / "plots_30" / "pehe_hypothesis_tests.csv"
TARGET_TUNERS = ["bayes", "random"]


def load_raw_results(results_folder: Path) -> pd.DataFrame:
    frames = []

    for raw_path in sorted(results_folder.glob("*/*/raw_results.csv")):
        learner = raw_path.parent.parent.name
        dimension = raw_path.parent.name
        df = pd.read_csv(raw_path)
        df["learner"] = learner
        df["dimension"] = dimension
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No raw_results.csv files found under {results_folder}")

    return pd.concat(frames, ignore_index=True)


def run_tests(all_df: pd.DataFrame) -> pd.DataFrame:
    filtered = all_df[all_df["tuner"].isin(TARGET_TUNERS)].copy()
    results = []

    for (learner, dimension), subset in filtered.groupby(["learner", "dimension"]):
        bayes = (
            subset[subset["tuner"] == "bayes"][["rep", "pehe"]]
            .rename(columns={"pehe": "pehe_bayes"})
            .sort_values("rep")
        )
        random = (
            subset[subset["tuner"] == "random"][["rep", "pehe"]]
            .rename(columns={"pehe": "pehe_random"})
            .sort_values("rep")
        )

        paired = bayes.merge(random, on="rep", how="inner").sort_values("rep")
        if paired.empty:
            continue

        ks_result = ks_2samp(
            paired["pehe_bayes"],
            paired["pehe_random"],
            alternative="two-sided",
            method="auto",
        )
        t_result = ttest_rel(paired["pehe_bayes"], paired["pehe_random"])

        diff = paired["pehe_bayes"] - paired["pehe_random"]

        results.append(
            {
                "learner": learner,
                "dimension": dimension,
                "n_pairs": len(paired),
                "bayes_mean_pehe": paired["pehe_bayes"].mean(),
                "random_mean_pehe": paired["pehe_random"].mean(),
                "mean_difference_bayes_minus_random": diff.mean(),
                "ks_statistic": ks_result.statistic,
                "ks_pvalue": ks_result.pvalue,
                "paired_t_statistic": t_result.statistic,
                "paired_t_pvalue": t_result.pvalue,
            }
        )

    return pd.DataFrame(results).sort_values(["learner", "dimension"]).reset_index(drop=True)


def main() -> None:
    all_df = load_raw_results(RESULTS_FOLDER)
    summary = run_tests(all_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
