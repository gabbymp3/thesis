import os
import importlib
import pandas as pd
import warnings

from src.experiment import run_experiment
from src.dgp import simulate_dataset

warnings.filterwarnings("ignore")


# -------------------------------------------------
# Select experiments
# -------------------------------------------------

CONFIGS_TO_RUN = [
    "config_1d",
    "config_2d",
    "config_4d",
    "config_6d",
]

def load_existing_results(raw_path):
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        completed = set(df["rep"].unique())
        print(f"    Found existing results with {len(completed)} completed reps")
        return df, completed
    else:
        return pd.DataFrame(), set()


# -------------------------------------------------
# Run experiments
# -------------------------------------------------

for config_name in CONFIGS_TO_RUN:
    print("\n" + "="*80)
    print(f"\nRunning experiment: {config_name}")
    print("="*80)

    module = importlib.import_module(
        f"src.experiment_configs.{config_name}"
    )

    config = module.get_config()

    for learner in config["learners"]:

        print(f"  → Learner: {learner['name']}")
        print("-"*80)
        # -------------------------------------------------
        # Output directory
        # -------------------------------------------------

        output_dir = os.path.join(
            f"results_R_{config['R']}",
            learner["name"],
            config["name"],
        )
        os.makedirs(output_dir, exist_ok=True)

        raw_path = os.path.join(output_dir, "raw_results.csv")

        # -------------------------------------------------
        # Load existing results
        # -------------------------------------------------

        existing_df, completed_reps = load_existing_results(raw_path)

        # -------------------------------------------------
        # Run experiment (only missing reps if found)
        # -------------------------------------------------

        summary, raw_results = run_experiment(
            learner_config={
                "name": learner["name"],
                "models": learner["models"],
                "propensity_model": learner["propensity_model"],
            },
            tuners=learner["tuners"],
            R=config["R"],
            simulate_dataset_fn=simulate_dataset,
            dgp_params=config["dgp_params"],
            base_seed=config["base_seed"],
            output_dir=output_dir,
            completed_reps=completed_reps
        )

        # -------------------------------------------------
        # Convert to df
        # -------------------------------------------------

        new_rows = []
        for tuner_name, metrics in raw_results.items():
            for r, pehe, pehe_plug in zip(
                metrics["rep"],
                metrics["pehe"],
                metrics["pehe_plug"]
            ):

                if r in completed_reps:
                    continue  # avoid duplicates

                new_rows.append({
                    "learner": learner["name"],
                    "tuner": tuner_name,
                    "rep": r,
                    "pehe": pehe,
                    "pehe_plug": pehe_plug,
                })

            new_df = pd.DataFrame(new_rows)

            # -------------------------------------------------
            # Append + save 
            # -------------------------------------------------

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(raw_path, index=False)
            print(f"    Saved raw results ({len(combined_df)} rows)")

            summary_df = pd.DataFrame(summary)
            summary_path = os.path.join(output_dir, "summary.csv")
            summary_df.to_csv(summary_path, index=False)

            print(f"  Saved results of {learner['name']} to {output_dir}")

    print(f"Experiment {config_name} completed.")
