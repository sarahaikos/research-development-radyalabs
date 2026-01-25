# Rule-Based Model

### Core components

1. **Data Structures** (`module_recommendation_engine.py`)
   - `AssessmentAttempt`: Represents assessment attempts per skill-level
   - `LearningActivity`: Represents learning activity logs
   - `Rule`: Represents recommendation rules with conditions and weights
   - `Module`: Represents learning modules with prerequisites
   - `Learner`: Represents learner with all their data

2. **Mastery Calculator** (`MasteryCalculator` class)
   - Computes attempt weights: `w_k = exp(-λ (N - k))`
   - Computes weighted score: `s_hat = sum(w_k * s_k) / sum(w_k)`
   - Computes mastery confidence: `M[u,s,l] = 1 / (1 + exp(-β_l (s_hat - θ_l)))`

3. **Rule Engine** (`RuleEngine` class)
   - Extracts rule conditions: `C_i ← normalize(relevant learner signals)`
   - Computes rule activation: `A_r = product(C_i)`
   - Applies rule weight: `S_r = A_r * W_r`

4. **Module Scorer** (`ModuleScorer` class)
   - Aggregates rule scores: `S_m = sum(S_r for all r recommending m)`
   - Applies hard constraints (prerequisites, time feasibility)

5. **Recommendation Engine** (`RecommendationEngine` class)
   - Orchestrates full pipeline
   - Generates explanations with contributing rules and signals
   - Returns ranked module recommendations

### Supporting Components

6. **Data Loader** (`data_loader.py`)
   - Loads modules from CSV
   - Loads module-skill mappings
   - Generates default recommendation rules
   - Creates sample learners for testing

7. **Example Usage** (`example_usage.py`)
   - Complete end-to-end example
   - Demonstrates all pipeline steps
   - Shows output format and explanations

## Pipeline Architecture

### Input
- **Learner u**: Learner identifier and profile
- **Assessment attempts**: Per skill-level assessment scores with timestamps
- **Learning activity logs**: Activity engagement data
- **Rule set R**: Recommendation rules with conditions and weights
- **Module set M**: Available learning modules with prerequisites

### Processing Steps

#### 1. Mastery Confidence Calculation
For each skill s and level l:

```
w_k = exp(-λ (N - k))                    # Attempt weights
s_hat = sum(w_k * s_k) / sum(w_k)        # Weighted score
M[u,s,l] = 1 / (1 + exp(-β_l (s_hat - θ_l)))  # Mastery confidence
```

#### 2. Rule Evaluation
For each rule r in R:

```
C_i ← normalize(relevant learner signals)  # Extract conditions
A_r = product(C_i)                          # Rule activation
S_r = A_r * W_r                             # Rule score
```

#### 3. Module Scoring
For each module m in M:

```
S_m = sum(S_r for all r recommending m)     # Aggregate scores
Remove m if prerequisites not met or time infeasible  # Apply constraints
```

#### 4. Selection
```
m* = module with highest S_m
```

### Output
- **Recommended module list**: Ranked by score
- **Explanation**: Contributing rules and key signals

### Pipeline

```
INPUT: Learner u, Assessment attempts, Activity logs, Rule set R, Module set M
  ↓
FOR each skill s and level l:
  - Compute attempt weights: w_k = exp(-λ (N - k))
  - Compute weighted score: s_hat = sum(w_k * s_k) / sum(w_k)
  - Compute mastery confidence: M[u,s,l] = 1 / (1 + exp(-β_l (s_hat - θ_l)))
  ↓
FOR each rule r in R:
  - Extract rule conditions: C_i ← normalize(relevant learner signals)
  - Compute rule activation: A_r = product(C_i)
  - Apply rule weight: S_r = A_r * W_r
  ↓
FOR each module m in M:
  - Aggregate rule scores: S_m = sum(S_r for all r recommending m)
  - Apply hard constraints: Remove m if prerequisites not met or time infeasible
  ↓
SELECT: m* = module with highest S_m
  ↓
OUTPUT: Recommended module list (ranked) + Explanation
```


## Configuration Parameters

### Mastery Calculation
- `lambda_decay` (λ): Decay parameter for attempt weights (default: 0.1)
- `beta_levels`: Dict mapping level to β parameter for sigmoid (default: {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25, 5: 0.3, 6: 0.35, 7: 0.4})
- `theta_levels`: Dict mapping level to θ threshold (default: {1: 50, 2: 55, 3: 60, 4: 65, 5: 70, 6: 75, 7: 80})

### Rule Weights
- Skill gap rules: 1.0
- Progression rules: 1.2
- Activity engagement: 0.8
- Prerequisite met: 1.1

## Rule Types

1. **Skill Gap Rules**: Recommend modules when mastery is below threshold
2. **Progression Rules**: Recommend next-level modules when current level is mastered
3. **Activity Engagement Rules**: Recommend based on learning activity patterns
4. **Prerequisite Rules**: Recommend modules when prerequisites are satisfied

## Data Requirements

### 1. Assessment Attempts (`sample_learner_assessments.csv`)

Contains assessment attempts for each skill-level combination.

**Columns:**
- `learner_id`: Unique learner identifier (string)
- `skill_code`: SFIA skill code (e.g., 'DAAN', 'DATS', 'PROG')
- `level`: Skill level (integer, 1-7)
- `score`: Assessment score (float, 0-100)
- `timestamp`: Timestamp in format 'YYYY-MM-DD HH:MM:SS'
- `attempt_number`: Attempt number for this skill-level (integer, 1-indexed)

**Example:**
```csv
learner_id,skill_code,level,score,timestamp,attempt_number
learner_001,DAAN,1,45,2024-01-15 10:30:00,1
learner_001,DAAN,1,52,2024-01-20 14:15:00,2
learner_001,DAAN,2,55,2024-02-01 11:20:00,1
```

### 2. Learning Activities (`sample_learner_activities.csv`)

Contains learning activity logs.

**Columns:**
- `learner_id`: Unique learner identifier (string)
- `activity_type`: Type of activity (string: 'video', 'quiz', 'exercise', 'project', 'reading')
- `module_id`: Module ID if applicable (string, can be empty)
- `skill_code`: SFIA skill code if applicable (string, can be empty)
- `duration_minutes`: Duration in minutes (float)
- `timestamp`: Timestamp in format 'YYYY-MM-DD HH:MM:SS'
- `completion_rate`: Completion rate (float, 0-1)

**Example:**
```csv
learner_id,activity_type,module_id,skill_code,duration_minutes,timestamp,completion_rate
learner_001,video,DATA_L2_01,DAAN,45,2024-01-15 08:00:00,0.95
learner_001,quiz,DATA_L2_01,DAAN,15,2024-01-15 09:00:00,1.0
learner_001,exercise,DATA_L2_02,DATS,60,2024-01-16 10:00:00,0.85
```

### 3. Learner Profile (`sample_learner_profile.csv`)

Contains learner profile information.

**Columns:**
- `learner_id`: Unique learner identifier (string)
- `available_time_hours`: Available time per week in hours (float)
- `completed_modules`: Comma-separated list of completed module IDs (string)

**Example:**
```csv
learner_id,available_time_hours,completed_modules
learner_001,12.5,"DATA_L2_01,SOFT_L2_01,DATA_L2_02"
```

## Output Format

Each recommendation includes:

```python
{
    'module_id': 'DATA_L3_01',
    'module_name': 'Data Analysis with Python or R',
    'score': 2.3456,
    'explanation': {
        'contributing_rules': [
            {
                'rule_id': 'gap_DATA_L3_01_DAAN_2',
                'rule_name': 'Skill Gap: DAAN Level 2',
                'rule_description': '...',
                'score': 0.8234
            }
        ],
        'key_signals': {
            'DAAN_2': 0.456,
            'DATS_2': 0.389,
            ...
        }
    }
}
```

## Extending the System

### Adding Custom Rules

```python
custom_rule = Rule(
    rule_id='custom_001',
    name='Custom Rule',
    conditions={
        'min_mastery': {
            'skill': 'DAAN',
            'level': 3,
            'min_mastery': 0.7
        }
    },
    weight=1.5,
    recommended_modules=['DATA_L4_01'],
    description='Custom rule description'
)

rules.append(custom_rule)
```

### Custom Condition Types

Extend `RuleEngine.extract_rule_conditions()` to handle new condition types.

## Performance Considerations

- Mastery calculation: O(S × L × A) where S=skills, L=levels, A=attempts
- Rule evaluation: O(R × C) where R=rules, C=conditions per rule
- Module scoring: O(M × R) where M=modules, R=rules
- Overall complexity: O(S×L×A + R×C + M×R)
