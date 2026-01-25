import csv
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
import random

from module_recommendation_engine import Module, Rule, AssessmentAttempt, LearningActivity


def load_modules_from_csv(csv_path: str) -> List[Module]:
    modules = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            module = Module(
                module_id=row['module_id'],
                module_name=row['module_name'],
                tag=row['tag'],
                sfia_level=int(row['sfia_level']),
                prerequisites=[],
                estimated_hours=estimate_module_hours(int(row['sfia_level'])),
                skills_covered=[]
            )
            modules.append(module)
    
    return modules


def load_module_mappings(csv_path: str) -> Dict[str, List[tuple]]:
    mappings = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            module_id = row['module_id']
            skill_code = row['sfia_skill_code']
            required_level = int(row['required_level'])
            achieved_level = int(row['achieved_level'])
            
            if module_id not in mappings:
                mappings[module_id] = []
            
            mappings[module_id].append((skill_code, required_level, achieved_level))
    
    return mappings


def estimate_module_hours(sfia_level: int) -> float:
    level_hours = {
        1: 2.0,
        2: 4.0,
        3: 8.0,
        4: 12.0,
        5: 16.0,
        6: 20.0,
        7: 24.0
    }
    return level_hours.get(sfia_level, 8.0)


def populate_module_data(modules: List[Module], mappings: Dict[str, List[tuple]]):
    for module in modules:
        if module.module_id in mappings:
            skills = mappings[module.module_id]
            module.skills_covered = skills
            
            # set prerequisites: need skills at required_level - 1
            prerequisites = []
            for skill_code, required_level, _ in skills:
                if required_level > 1:
                    prerequisites.append((skill_code, required_level - 1))
            module.prerequisites = list(set(prerequisites))


def generate_default_rules(modules: List[Module], mappings: Dict[str, List[tuple]]) -> List[Rule]:
    rules = []
    
    # rule 1: recommend modules for skill gaps (low mastery)
    rule_id = 1
    for module in modules:
        if not module.skills_covered:
            continue  
        # create a rule that recommends this module when there's a skill gap
        for skill_code, required_level, achieved_level in module.skills_covered[:3]:
            rule = Rule(
                rule_id=f'gap_{module.module_id}_{skill_code}_{required_level}',
                name=f'Skill Gap: {skill_code} Level {required_level}',
                conditions={
                    'skill_gap': {
                        'skill': skill_code,
                        'level': required_level,
                        'max_mastery': 0.5
                    }
                },
                weight=1.0,
                recommended_modules=[module.module_id],
                description=f'Recommend {module.module_name} to address gap in {skill_code} at level {required_level}'
            )
            rules.append(rule)
            rule_id += 1
    
    # eule 2: recommend next level modules when current level is mastered
    for module in modules:
        if not module.skills_covered:
            continue
        # check if this is a progression module (next level)
        for skill_code, required_level, achieved_level in module.skills_covered[:2]:
            if required_level > 1:
                rule = Rule(
                    rule_id=f'progression_{module.module_id}_{skill_code}_{required_level}',
                    name=f'Level Progression: {skill_code} to Level {required_level}',
                    conditions={
                        'level_progression': {
                            'skill': skill_code,
                            'current_level': required_level - 1
                        },
                        'min_mastery': {
                            'skill': skill_code,
                            'level': required_level - 1,
                            'min_mastery': 0.7
                        }
                    },
                    weight=1.2,
                    recommended_modules=[module.module_id],
                    description=f'Recommend {module.module_name} for progression in {skill_code} to level {achieved_level}'
                )
                rules.append(rule)
                rule_id += 1
    
    # rule 3: recommend modules based on activity engagement
    for module in modules:
        if module.tag in ['DATA', 'SOFT', 'DEVO']:
            rule = Rule(
                rule_id=f'activity_{module.module_id}',
                name=f'Activity Engagement: {module.tag}',
                conditions={
                    'activity_type': {
                        'type': 'exercise',
                        'min_count': 5
                    }
                },
                weight=0.8,
                recommended_modules=[module.module_id],
                description=f'Recommend {module.module_name} based on high engagement with {module.tag} activities'
            )
            rules.append(rule)
            rule_id += 1
    
    # rule 4: recommend modules that build on completed modules
    for module in modules:
        if not module.prerequisites:
            continue
        # create rule that recommends this module if prerequisites are met
        prereq_conditions = {}
        for skill_code, min_level in module.prerequisites[:2]:
            prereq_conditions[f'prereq_{skill_code}_{min_level}'] = {
                'skill': skill_code,
                'level': min_level,
                'min_mastery': 0.5
            }
        
        if prereq_conditions:
            rule = Rule(
                rule_id=f'prereq_{module.module_id}',
                name=f'Prerequisite Met: {module.module_name}',
                conditions={
                    'min_mastery': {
                        'skill': module.prerequisites[0][0],
                        'level': module.prerequisites[0][1],
                        'min_mastery': 0.5
                    }
                },
                weight=1.1,
                recommended_modules=[module.module_id],
                description=f'Recommend {module.module_name} as prerequisites are met'
            )
            rules.append(rule)
            rule_id += 1
    
    return rules


def create_sample_learner(learner_id: str, skill_codes: List[str], 
                         n_attempts_per_skill: int = 3) -> 'Learner':
    from module_recommendation_engine import Learner
    learner = Learner(learner_id=learner_id)
    base_time = datetime.now() - timedelta(days=90)
    
    for skill_code in skill_codes:
        for level in range(1, 6):
            for attempt_num in range(1, n_attempts_per_skill + 1):
                base_score = 40 + (level - 1) * 10
                improvement = (attempt_num - 1) * 8
                score = min(95, base_score + improvement + random.randint(-5, 5))
                
                attempt = AssessmentAttempt(
                    skill_code=skill_code,
                    level=level,
                    score=score,
                    timestamp=base_time + timedelta(
                        days=random.randint(0, 60),
                        hours=random.randint(0, 23)
                    ),
                    attempt_number=attempt_num
                )
                learner.assessment_attempts.append(attempt)
    
    activity_types = ['video', 'quiz', 'exercise', 'project', 'reading']
    
    for _ in range(50): 
        activity = LearningActivity(
            activity_type=random.choice(activity_types),
            module_id=None,
            skill_code=random.choice(skill_codes) if skill_codes else None,
            duration_minutes=random.uniform(10, 120),
            timestamp=base_time + timedelta(
                days=random.randint(0, 90),
                hours=random.randint(0, 23)
            ),
            completion_rate=random.uniform(0.5, 1.0)
        )
        learner.learning_activities.append(activity)
    learner.available_time_hours = random.uniform(5, 20)    
    return learner


def load_learner_from_csv(learner_id: str, assessments_csv: str, 
                         activities_csv: str, profile_csv: str = None) -> 'Learner':
    from module_recommendation_engine import Learner, AssessmentAttempt, LearningActivity
    
    learner = Learner(learner_id=learner_id)
    
    # assessment attempts
    with open(assessments_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['learner_id'] == learner_id:
                attempt = AssessmentAttempt(
                    skill_code=row['skill_code'],
                    level=int(row['level']),
                    score=float(row['score']),
                    timestamp=datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),
                    attempt_number=int(row['attempt_number'])
                )
                learner.assessment_attempts.append(attempt)
    
    # learning activities
    with open(activities_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['learner_id'] == learner_id:
                activity = LearningActivity(
                    activity_type=row['activity_type'],
                    module_id=row['module_id'] if row['module_id'] else None,
                    skill_code=row['skill_code'] if row['skill_code'] else None,
                    duration_minutes=float(row['duration_minutes']),
                    timestamp=datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S'),
                    completion_rate=float(row['completion_rate'])
                )
                learner.learning_activities.append(activity)
    
    # profile if provided
    if profile_csv and os.path.exists(profile_csv):
        with open(profile_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['learner_id'] == learner_id:
                    learner.available_time_hours = float(row.get('available_time_hours', 0))
                    completed = row.get('completed_modules', '')
                    if completed:
                        learner.completed_modules = set(completed.split(','))
    
    return learner