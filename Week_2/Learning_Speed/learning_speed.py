import json
import os
from datetime import datetime
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated",
    category=FutureWarning,
)

# Step 1: Input Data
with open("../Sample_Data/learner_evidence.json") as f:
    learner_data = json.load(f)

try:
    with open("../Rule_Based/learner_recommendations.json") as f:
        learner_recs = json.load(f)
except FileNotFoundError:
    learner_recs = {}


# Step 2: Flatten events for analysis
progression_records = []

for learner_id, learner_info in learner_data.items():
    for cap_id, cap_data in learner_info["capabilities"].items():
        events = cap_data.get("learning_events", [])
        events_sorted = sorted(events, key=lambda x: x["timestamp"])

        for event in events_sorted:
            timestamp = datetime.fromisoformat(event["timestamp"])
            score = event["score"]
            progression_records.append(
                {
                    "Learner": learner_id,
                    "Capability": cap_id,
                    "EventID": event.get("event_id"),
                    "Attempt": event["attempt"],
                    "Score": score,
                    "Timestamp": timestamp,
                }
            )

progression_df = pd.DataFrame(progression_records)

# Step 3: Sort and compute time differences
progression_df.sort_values(
    by=["Learner", "Capability", "Timestamp"], inplace=True
)
progression_df["TimeDiffDays"] = (
    progression_df.groupby(["Learner", "Capability"])["Timestamp"]
    .diff()
    .dt.days.fillna(0)
)

progression_df["ScoreDelta"] = (
    progression_df.groupby(["Learner", "Capability"])["Score"]
    .diff()
    .fillna(0)
)
progression_df["IntervalMasteryRate"] = progression_df["ScoreDelta"] / (
    progression_df["TimeDiffDays"].replace(0, 1)
)

# Step 4: Handle shared events across capabilities
event_cap_counts = (
    progression_df.groupby("EventID")["Capability"].nunique()
)
shared_event_ids = event_cap_counts[event_cap_counts > 1].index
progression_df["SharedEvent"] = progression_df["EventID"].isin(
    shared_event_ids
)

# Step 5 & 6: Compute raw and weighted progression metrics
def compute_group_metrics(group: pd.DataFrame) -> pd.Series:
    group = group.sort_values("Timestamp")
    first_score = group["Score"].iloc[0]
    last_score = group["Score"].iloc[-1]

    score_improvement = last_score - first_score
    total_attempts = len(group)

    total_time_days = (
        group["Timestamp"].iloc[-1] - group["Timestamp"].iloc[0]
    ).days
    total_time_days = max(total_time_days, 1)

    n = total_attempts
    weights = pd.Series(
        range(1, n + 1), index=group.index, dtype=float
    )
    weights = weights / weights.sum()
    weighted_score = (group["Score"] * weights).sum()
    weighted_score_improvement = weighted_score - first_score

    mastery_rate = score_improvement / total_time_days
    weighted_mastery_rate = weighted_score_improvement / total_time_days

    avg_time_between_attempts = group["TimeDiffDays"].mean()
    avg_interval_mastery_rate = group["IntervalMasteryRate"].replace(
        [float("inf"), float("-inf")], 0
    ).mean()

    return pd.Series(
        {
            "TotalAttempts": total_attempts,
            "TotalTimeDays": total_time_days,
            "ScoreImprovement": score_improvement,
            "MasteryRate": mastery_rate,
            "WeightedMasteryRate": weighted_mastery_rate,
            "AvgTimeBetweenAttempts": avg_time_between_attempts,
            "AvgIntervalMasteryRate": avg_interval_mastery_rate,
        }
    )

summary_df = (
    progression_df.groupby(["Learner", "Capability"])
    .apply(compute_group_metrics)
    .reset_index()
)

# Step 8: Visualizations
learners = progression_df["Learner"].unique()

# Plot 1: Score progression over time (annotate attempts, highlight shared)
for learner_id in learners:
    learner_df = progression_df[progression_df["Learner"] == learner_id]
    capabilities = learner_df["Capability"].unique()

    plt.figure(figsize=(10, 6))
    for cap_id in capabilities:
        cap_df = learner_df[learner_df["Capability"] == cap_id]
        plt.plot(
            cap_df["Timestamp"],
            cap_df["Score"],
            marker="o",
            label=cap_id,
        )
        for _, row in cap_df.iterrows():
            plt.text(
                row["Timestamp"],
                row["Score"] + 1,
                f"A{int(row['Attempt'])}",
                ha="center",
                fontsize=8,
            )
        # red circle -> shared event
        shared_cap_df = cap_df[cap_df["SharedEvent"]]
        if not shared_cap_df.empty:
            plt.scatter(
                shared_cap_df["Timestamp"],
                shared_cap_df["Score"],
                s=80,
                facecolors="none",
                edgecolors="red",
                linewidths=2,
            )

    plt.title(f"Score Progression Over Time - {learner_id}")
    plt.xlabel("Date")
    plt.ylabel("Score")
    plt.ylim(0, 100)
    plt.legend(title="Capability", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"score_progression_{learner_id}.png"))
    plt.close()

# Plot 2: Score improvement per attempt
for learner_id in learners:
    learner_df = progression_df[progression_df["Learner"] == learner_id]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=learner_df,
        x="Attempt",
        y="Score",
        hue="Capability",
        marker="o",
        palette="tab10",
    )
    plt.title(f"Score Improvement Per Attempt - {learner_id}")
    plt.xlabel("Attempt")
    plt.ylabel("Score")
    plt.ylim(0, 100)
    plt.xticks(sorted(learner_df["Attempt"].unique()))
    plt.legend(title="Capability", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"score_improvement_{learner_id}.png"))
    plt.close()

# Plot 3: Effect of time on progression (scatter: TimeDiffDays vs ScoreDelta)
for learner_id in learners:
    learner_df = progression_df[progression_df["Learner"] == learner_id]

    interval_df = learner_df[learner_df["Attempt"] > learner_df["Attempt"].min()]

    if interval_df.empty:
        continue

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=interval_df,
        x="TimeDiffDays",
        y="ScoreDelta",
        hue="Capability",
        palette="tab10",
    )
    plt.title(f"Effect of Time Between Attempts on Score Change - {learner_id}")
    plt.xlabel("TimeDiffDays (days between attempts)")
    plt.ylabel("ScoreDelta (score gain from previous attempt)")
    plt.axhline(0, color="grey", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"time_vs_score_change_{learner_id}.png",
        )
    )
    plt.close()

# Plot 4: MasteryRate per Capability
for learner_id in learners:
    learner_summary = summary_df[summary_df["Learner"] == learner_id]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=learner_summary,
        x="Capability",
        y="MasteryRate",
    )
    plt.title(f"Mastery Rate per Capability - {learner_id}")
    plt.xlabel("Capability")
    plt.ylabel("MasteryRate (points per day)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"mastery_rate_per_capability_{learner_id}.png")
    )
    plt.close()

# Plot 5: WeightedMasteryRate per Capability
for learner_id in learners:
    learner_summary = summary_df[summary_df["Learner"] == learner_id]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=learner_summary,
        x="Capability",
        y="WeightedMasteryRate",
    )
    plt.title(f"Weighted Mastery Rate per Capability - {learner_id}")
    plt.xlabel("Capability")
    plt.ylabel("WeightedMasteryRate (points per day)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_dir,
            f"weighted_mastery_rate_per_capability_{learner_id}.png",
        )
    )
    plt.close()

# Step 7 & 9: Store metrics for output (CSV summary)
summary_df.to_csv(
    os.path.join(
        output_dir,
        "learning_speed_summary.csv",
    ),
    index=False,
)

# Step 10: Simple analysis (identify relatively fast/slow learners)
learner_speed = (
    summary_df.groupby("Learner")[["MasteryRate", "WeightedMasteryRate"]]
    .mean()
    .reset_index()
    .sort_values("MasteryRate", ascending=False)
)

print("\nAverage progression speed per learner (points per day):")
print(learner_speed.to_string(index=False))