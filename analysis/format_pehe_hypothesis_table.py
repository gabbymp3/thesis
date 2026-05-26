from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "plots_30" / "pehe_hypothesis_tests.csv"
OUTPUT_DIR = ROOT / "plots_30" / "tables"
CSV_OUTPUT = OUTPUT_DIR / "pehe_hypothesis_tests_thesis.csv"
LATEX_OUTPUT = OUTPUT_DIR / "pehe_hypothesis_tests_thesis.tex"
MARKDOWN_OUTPUT = OUTPUT_DIR / "pehe_hypothesis_tests_thesis.md"


def sig_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def verdict(diff: float, p: float) -> str:
    if pd.isna(p):
        return "No difference detected"
    if p < 0.05 and diff < 0:
        return "Bayes lower"
    if p < 0.05 and diff > 0:
        return "Random lower"
    return "No significant difference"


def format_pvalue(p: float) -> str:
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def build_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["learner"] = out["learner"].str.upper()
    out["dimension"] = out["dimension"].str.upper()
    out["bayes_mean_pehe"] = out["bayes_mean_pehe"].round(3)
    out["random_mean_pehe"] = out["random_mean_pehe"].round(3)
    out["mean_difference_bayes_minus_random"] = out["mean_difference_bayes_minus_random"].round(3)
    out["ks_statistic"] = out["ks_statistic"].round(3)
    out["paired_t_statistic"] = out["paired_t_statistic"].round(3)
    out["ks_pvalue_fmt"] = out["ks_pvalue"].map(format_pvalue)
    out["paired_t_pvalue_fmt"] = out["paired_t_pvalue"].map(format_pvalue)
    out["paired_t_sig"] = out["paired_t_pvalue"].map(sig_stars)
    out["interpretation"] = [
        verdict(diff, p)
        for diff, p in zip(out["mean_difference_bayes_minus_random"], out["paired_t_pvalue"])
    ]

    out = out.rename(
        columns={
            "learner": "Learner",
            "dimension": "Setting",
            "n_pairs": "N",
            "bayes_mean_pehe": "Bayes mean",
            "random_mean_pehe": "Random mean",
            "mean_difference_bayes_minus_random": "Mean diff (B-R)",
            "ks_statistic": "KS stat",
            "ks_pvalue_fmt": "KS p",
            "paired_t_statistic": "Paired t",
            "paired_t_pvalue_fmt": "Paired t p",
            "paired_t_sig": "Sig.",
            "interpretation": "Interpretation",
        }
    )

    return out[
        [
            "Learner",
            "Setting",
            "N",
            "Bayes mean",
            "Random mean",
            "Mean diff (B-R)",
            "KS stat",
            "KS p",
            "Paired t",
            "Paired t p",
            "Sig.",
            "Interpretation",
        ]
    ]


def write_latex(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    header = " & ".join(columns) + r" \\"
    rows = [" & ".join(str(row[col]) for col in columns) + r" \\" for _, row in df.iterrows()]
    latex = "\n".join(
        [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Distributional and paired-mean comparisons of PEHE between Bayesian and random tuning across 30 Monte Carlo replications. Negative mean differences favor Bayes.}",
            r"\label{tab:pehe_hypothesis_tests}",
            r"\begin{tabular}{llrcccccccll}",
            r"\hline",
            header,
            r"\hline",
            *rows,
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text(latex)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in columns) + " |"
        for _, row in df.iterrows()
    ]
    path.write_text("\n".join([header, divider, *rows, ""]))


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    thesis_df = build_table(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thesis_df.to_csv(CSV_OUTPUT, index=False)
    write_latex(thesis_df, LATEX_OUTPUT)
    write_markdown(thesis_df, MARKDOWN_OUTPUT)

    print(thesis_df.to_string(index=False))
    print(f"\nSaved: {CSV_OUTPUT}")
    print(f"Saved: {LATEX_OUTPUT}")
    print(f"Saved: {MARKDOWN_OUTPUT}")


if __name__ == "__main__":
    main()
