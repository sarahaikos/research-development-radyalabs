import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_learning_speed_summary() -> pd.DataFrame:
    summary_path = "../Learning_Speed/visualizations/learning_speed_summary.csv"
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Learning speed summary not found at {summary_path}. "
            "Run Learning_Speed/learning_speed.py first."
        )
    df = pd.read_csv(summary_path)
    return df


def load_rule_based_recommendations() -> dict:
    recs_path = "../Rule_Based/learner_recommendations.json"
    if not os.path.exists(recs_path):
        raise FileNotFoundError(
            f"Rule-based recommendations not found at {recs_path}. "
            "Run Rule_Based/model.py first."
        )
    with open(recs_path) as f:
        return json.load(f)


def build_mastery_dataframe(learner_recs: dict) -> pd.DataFrame:
    rows = []
    for learner_id, info in learner_recs.items():
        mastery = info.get("mastery", {})
        for cap_id, cap_info in mastery.items():
            rows.append(
                {
                    "Learner": learner_id,
                    "Capability": cap_id,
                    "MasteryState": cap_info.get("state"),
                    "MasteryScore": cap_info.get("score"),
                    "Level": cap_info.get("level"),
                }
            )
    return pd.DataFrame(rows)


def merge_speed_and_mastery(
    speed_df: pd.DataFrame, mastery_df: pd.DataFrame
) -> pd.DataFrame:
    merged = pd.merge(
        mastery_df,
        speed_df,
        on=["Learner", "Capability"],
        how="outer",
        indicator=True,
    )
    return merged


def analyze_connections(merged_df: pd.DataFrame) -> None:
    print("\n=== Basic merged stats (per capability) ===")
    print(
        merged_df[
            [
                "Learner",
                "Capability",
                "MasteryState",
                "MasteryScore",
                "TotalAttempts",
                "TotalTimeDays",
                "MasteryRate",
                "WeightedMasteryRate",
                "AvgIntervalMasteryRate",
            ]
        ].head(20).to_string(index=False)
    )

    numeric_cols = [
        "MasteryScore",
        "MasteryRate",
        "WeightedMasteryRate",
        "AvgIntervalMasteryRate",
    ]
    corr_df = merged_df[numeric_cols].dropna()
    if not corr_df.empty:
        print("\n=== Correlation between mastery and progression metrics ===")
        print(corr_df.corr().to_string())
    else:
        print(
            "\nNot enough numeric data to compute correlations "
            "between mastery and progression metrics."
        )

    learner_agg = (
        merged_df.groupby("Learner")[
            ["MasteryScore", "MasteryRate", "WeightedMasteryRate", "AvgIntervalMasteryRate"]
        ]
        .mean()
        .reset_index()
    )
    print("\n=== Average mastery and speed metrics per learner ===")
    print(learner_agg.to_string(index=False))


def create_visualizations(merged_df: pd.DataFrame, output_dir: str) -> None:
    """
    Create visualizations that connect rule-based mastery with learning speed.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: MasteryScore vs MasteryRate (per capability)
    scatter_df = merged_df.dropna(
        subset=["MasteryScore", "MasteryRate"]
    )
    if not scatter_df.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=scatter_df,
            x="MasteryRate",
            y="MasteryScore",
            hue="Learner",
            style="Learner",
            s=80,
        )
        plt.title("Mastery Score vs Mastery Rate (per Capability)")
        plt.xlabel("MasteryRate (points per day)")
        plt.ylabel("MasteryScore (rule-based mastery)")
        plt.axvline(0, color="grey", linestyle="--", linewidth=1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "mastery_vs_speed_scatter.png"))
        plt.close()

    # Plot 2: WeightedMasteryRate vs MasteryScore
    weighted_df = merged_df.dropna(
        subset=["MasteryScore", "WeightedMasteryRate"]
    )
    if not weighted_df.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=weighted_df,
            x="WeightedMasteryRate",
            y="MasteryScore",
            hue="Learner",
            style="Learner",
            s=80,
        )
        plt.title("Mastery Score vs Weighted Mastery Rate (per Capability)")
        plt.xlabel("WeightedMasteryRate (points per day)")
        plt.ylabel("MasteryScore (rule-based mastery)")
        plt.axvline(0, color="grey", linestyle="--", linewidth=1)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                output_dir, "mastery_vs_weighted_speed_scatter.png"
            )
        )
        plt.close()

    # Plot 3: Learner-level comparison of average mastery and speed
    learner_agg = (
        merged_df.groupby("Learner")[
            ["MasteryScore", "MasteryRate", "WeightedMasteryRate"]
        ]
        .mean()
        .reset_index()
    )
    if not learner_agg.empty:
        plt.figure(figsize=(8, 6))
        learner_melt = learner_agg.melt(
            id_vars="Learner",
            value_vars=[
                "MasteryScore",
                "MasteryRate",
                "WeightedMasteryRate",
            ],
            var_name="Metric",
            value_name="Value",
        )
        sns.barplot(
            data=learner_melt,
            x="Learner",
            y="Value",
            hue="Metric",
        )
        plt.title("Average Mastery and Speed Metrics per Learner")
        plt.xlabel("Learner")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "learner_metrics_comparison.png"))
        plt.close()


def main() -> None:
    speed_df = load_learning_speed_summary()
    learner_recs = load_rule_based_recommendations()
    mastery_df = build_mastery_dataframe(learner_recs)

    merged_df = merge_speed_and_mastery(speed_df, mastery_df)
    merged_path = "learner_rule_based_speed_merged.csv"

    merged_df.to_csv(merged_path, index=False)
    print(f"\nMerged learner connection CSV written to: {merged_path}")

    analyze_connections(merged_df)

    visuals_dir = "visualizations"
    create_visualizations(merged_df, visuals_dir)

if __name__ == "__main__":
    main()