import json
import math

with open('../SFIA_Capability_Evidence/SFIA_Capabilities_Full.json') as f:
    sfia_data = json.load(f)

with open('../Sample_Data/learner_evidence.json') as f:
    learner_data = json.load(f)

with open('../Sample_Data/rules.json') as f:
    rules = json.load(f)

with open('../Sample_Data/modules.json') as f:
    modules = json.load(f)

with open('../Config/SFIA_Level_Threshold.json') as f:
    level_thresholds = json.load(f)["levels"]

lambda_decay = 0.5

capability_lookup = {}
for skill in sfia_data.values():
    for cap in skill["capabilities"]:
        capability_lookup[cap["id"]] = {
            "level": cap["level"],
            "description": cap["description"],
            "tag": skill.get("tag"),
            "sub_area": skill.get("sub_area"),
            "subcategory": skill.get("subcategory")
        }

def compute_mastery(scores, level):
    if not scores:
        return {
            "state": "Not Mastered",
            "score": 0.0,
            "level": level
        }
    N = len(scores)
    weights = [math.exp(-lambda_decay * (N - k - 1)) for k in range(N)]
    weighted_score = sum([w*s for w,s in zip(weights, scores)]) / sum(weights)
    thresholds = level_thresholds.get(str(level))
    if weighted_score >= thresholds["mastered"]:
        state = "Mastered"
    elif weighted_score >= thresholds["partial"]:
        state = "Partial"
    else:
        state = "Not Mastered"
    return {
        "state": state,
        "score": round(weighted_score, 2),
        "level": level
    }

all_recommendations = {}

for learner_id, learner in learner_data.items():
    # STEP 1: Compute mastery per capability
    observed_mastery = {}
    for cap_id, cap_data in learner["capabilities"].items():

        events = cap_data["learning_events"]
        scores = [e["score"] for e in events]

        sfia_cap = capability_lookup.get(cap_id)
        if not sfia_cap:
            continue 

        observed_mastery[cap_id] = compute_mastery(
            scores,
            sfia_cap["level"]
        )

    print(f"\nMastery for {learner_id}:")
    print(observed_mastery)

    # STEP 2: Evaluate rules dynamically
    # Soft Logic
    for rule in rules:
        satisfied = 0
        total = len(rule["conditions"])

        for cap_id, required_state in rule["conditions"].items():
            learner_state = observed_mastery.get(cap_id, {}).get("state")

            if required_state == "Mastered" and learner_state == "Mastered":
                satisfied += 1
            elif required_state == "Partial" and learner_state in ["Partial", "Mastered"]:
                satisfied += 1

        coverage = satisfied / total if total > 0 else 0
        rule["activated"] = coverage >= 0.5
        rule["score"] = rule["weight"] * coverage

    # STEP 3: Aggregate module scores
    module_scores = {}
    module_explanations = {}

    for module_name, info in modules.items():
        score = 0
        triggered_rules = []
        contributing_caps = []

        for rule in rules:
            if rule["module"] == module_name and rule["activated"]:
                score += rule["score"]
                triggered_rules.append(rule["name"])

                for cap_id in rule["conditions"]:
                    if cap_id in observed_mastery:
                        cap_meta = capability_lookup[cap_id]
                        contributing_caps.append({
                            "id": cap_id,
                            **cap_meta,
                            "state": observed_mastery[cap_id]["state"]
                        })

        if not info["available"]:
            score = 0

        module_scores[module_name] = score
        module_explanations[module_name] = {
            "rules": triggered_rules,
            "capabilities": contributing_caps
        }

    # STEP 4: Rank modules
    ranked = sorted(module_scores.items(), key=lambda x: x[1], reverse=True)

    all_recommendations[learner_id] = {
        "mastery": observed_mastery,
        "recommendations": [
            {
                "module": name,
                "score": round(score, 2),
                **module_explanations[name]
            }
            for name, score in ranked if score > 0
        ]
    }

with open('learner_recommendations.json', 'w') as f:
    json.dump(all_recommendations, f, indent=2)