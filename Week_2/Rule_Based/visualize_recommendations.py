import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

with open("learner_recommendations.json") as f:
    recommendations = json.load(f)

with open("../Sample_Data/learner_evidence.json") as f:
    learner_evidence = json.load(f)

with open('../SFIA_Capability_Evidence/SFIA_Capabilities_Full.json') as f:
    sfia_data = json.load(f)

capability_meta = {}
sfia_lookup = {}

for skill in sfia_data.values():
    for cap in skill["capabilities"]:
        capability_meta[cap["id"]] = {
            "Tag": skill.get("tag"),
            "Description": cap.get("description"),
            "Level": cap.get("level"),
            "Subcategory": skill.get("subcategory"),
            "Sub_area": skill.get("sub_area")
        }
        sfia_lookup[cap["id"]] = capability_meta[cap["id"]]

mastery_records = []

for learner_id, learner_data in recommendations.items():
    mastery = learner_data["mastery"]
    for cap_id, cap_data in mastery.items():
        meta = capability_meta.get(cap_id, {})
        mastery_records.append({
            "Learner": learner_id,
            "Capability": cap_id,
            "Level": cap_data["state"],
            "Tag": meta.get("Tag", "UNKNOWN")
        })

mastery_df = pd.DataFrame(mastery_records)

level_mapping = {"Not Mastered": 0, "Partial": 1, "Mastered": 2}

heatmap_df = mastery_df.assign(
    LevelNum=mastery_df["Level"].map(level_mapping)
).pivot_table(
    index="Capability",
    columns="Learner",
    values="LevelNum",
    aggfunc="max"
)

plt.figure(figsize=(10, len(heatmap_df)/2))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Mastery Level'})
plt.title("Learner Capability Mastery Heatmap")
plt.xlabel("Learner")
plt.ylabel("Capability")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "capability_mastery_heatmap.png"))
plt.close()

for learner_id, learner_data in recommendations.items():
    recs = learner_data["recommendations"]
    
    if not recs:
        continue

    modules = [r["module"] for r in recs]
    scores = [r["score"] for r in recs]

    plt.figure(figsize=(6,4))
    sns.barplot(x=scores, y=modules, color="mediumseagreen")
    plt.title(f"Module Recommendations for {learner_id}")
    plt.xlabel("Recommendation Score")
    plt.ylabel("Module")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"module_recommendations_{learner_id}.png"))
    plt.close()

tag_summary_records = []

for _, row in mastery_df.iterrows():
    tag_summary_records.append({
        "Learner": row["Learner"],
        "Tag": row["Tag"],
        "Level": row["Level"]
    })

tag_df = pd.DataFrame(tag_summary_records)
tag_summary = tag_df.groupby(['Learner','Tag','Level']).size().unstack(fill_value=0)

for learner_id in tag_summary.index.get_level_values(0).unique():
    df = tag_summary.loc[learner_id]
    df.plot(kind='bar', stacked=True, figsize=(8,5), colormap="Pastel1")
    plt.title(f"Tag-Level Mastery Summary for {learner_id}")
    plt.xlabel("Tag")
    plt.ylabel("Count of Capabilities")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"tag_mastery_summary_{learner_id}.png"))
    plt.close()

progression_records = []

for learner_id, learner_data in learner_evidence.items():
    for cap_id, cap_data in learner_data["capabilities"].items():
        events = sorted(
            cap_data["learning_events"],
            key=lambda x: x["timestamp"]
        )
        for e in events:
            progression_records.append({
                "Learner": learner_id,
                "Capability": cap_id,
                "Attempt": e["attempt"],
                "Score": e["score"],
                "Timestamp": pd.to_datetime(e["timestamp"])
            })

progression_df = pd.DataFrame(progression_records)
progression_df["MasterySignal"] = progression_df["Score"] / 100.0

progression_df["SmoothedMastery"] = progression_df.groupby(
    ["Learner","Capability"]
)["MasterySignal"].transform(lambda x: x.expanding().mean())

for learner_id in progression_df["Learner"].unique():
    learner_prog = progression_df[progression_df["Learner"] == learner_id]

    plt.figure(figsize=(10,5))
    sns.lineplot(
        data=learner_prog,
        x="Attempt",
        y="SmoothedMastery",
        hue="Capability",
        marker="o"
    )

    plt.title(f"Learning Progression for {learner_id}")
    plt.xlabel("Learning Attempt")
    plt.ylabel("Estimated Mastery")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"learning_progression_{learner_id}.png"))
    plt.close()