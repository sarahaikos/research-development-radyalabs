import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json


@dataclass
class AssessmentAttempt:
    skill_code: str
    level: int
    score: float  # 0-100
    timestamp: datetime
    attempt_number: int  # k-th attempt (1-indexed)


@dataclass
class LearningActivity:
    activity_type: str
    module_id: Optional[str]
    skill_code: Optional[str]
    duration_minutes: float
    timestamp: datetime
    completion_rate: float  # 0-1


@dataclass
class Rule:
    rule_id: str
    name: str
    conditions: Dict[str, any]  # conditions to check (e.g., {'skill': 'DAAN', 'min_level': 3})
    weight: float
    recommended_modules: List[str]  # module IDs this rule recommends
    description: str


@dataclass
class Module:
    module_id: str
    module_name: str
    tag: str
    sfia_level: int
    prerequisites: List[str]  # list of (skill_code, min_level) tuples
    estimated_hours: float
    skills_covered: List[Tuple[str, int, int]]  # (skill_code, required_level, achieved_level)


@dataclass
class Learner:
    learner_id: str
    assessment_attempts: List[AssessmentAttempt] = field(default_factory=list)
    learning_activities: List[LearningActivity] = field(default_factory=list)
    completed_modules: Set[str] = field(default_factory=set)
    available_time_hours: float = 0.0


class MasteryCalculator:
    def __init__(self, lambda_decay: float = 0.1, beta_levels: Dict[int, float] = None, 
                 theta_levels: Dict[int, float] = None):
        self.lambda_decay = lambda_decay
        self.beta_levels = beta_levels or {1: 0.1, 2: 0.15, 3: 0.2, 4: 0.25, 5: 0.3, 6: 0.35, 7: 0.4}
        self.theta_levels = theta_levels or {1: 50, 2: 55, 3: 60, 4: 65, 5: 70, 6: 75, 7: 80}
    
    def compute_attempt_weights(self, n_attempts: int) -> np.ndarray:
        if n_attempts == 0:
            return np.array([])
        k_values = np.arange(1, n_attempts + 1)
        weights = np.exp(-self.lambda_decay * (n_attempts - k_values))
        return weights
    
    def compute_weighted_score(self, attempts: List[AssessmentAttempt]) -> float:
        if not attempts:
            return 0.0
        
        sorted_attempts = sorted(attempts, key=lambda x: x.timestamp)
        n = len(sorted_attempts)
        weights = self.compute_attempt_weights(n)
        scores = np.array([a.score for a in sorted_attempts])
        
        weighted_sum = np.sum(weights * scores)
        weight_sum = np.sum(weights)
        
        if weight_sum == 0:
            return 0.0
        
        return weighted_sum / weight_sum
    
    def compute_mastery_confidence(self, skill_code: str, level: int, 
                                   attempts: List[AssessmentAttempt]) -> float:
        relevant_attempts = [a for a in attempts 
                           if a.skill_code == skill_code and a.level == level]
        
        if not relevant_attempts:
            return 0.0
        
        s_hat = self.compute_weighted_score(relevant_attempts)
        beta = self.beta_levels.get(level, 0.2)
        theta = self.theta_levels.get(level, 60)
        
        exponent = -beta * (s_hat - theta)
        mastery = 1.0 / (1.0 + np.exp(exponent))
        
        return mastery


class RuleEngine:
    def __init__(self):
        self.rules: List[Rule] = []
    
    def add_rule(self, rule: Rule):
        self.rules.append(rule)
    
    def normalize_signal(self, signal_value: float, signal_type: str) -> float:
        if signal_type == 'mastery':
            return max(0.0, min(1.0, signal_value))
        elif signal_type == 'activity_count':
            return max(0.0, min(1.0, signal_value / 100.0))
        elif signal_type == 'time_spent':
            return max(0.0, min(1.0, signal_value / 100.0))
        elif signal_type == 'score':
            return max(0.0, min(1.0, signal_value / 100.0))
        else:
            return 1.0 / (1.0 + np.exp(-signal_value))
    
    def extract_rule_conditions(self, rule: Rule, learner: Learner, 
                               mastery_matrix: Dict[Tuple[str, int], float],
                               activity_stats: Dict[str, float]) -> List[float]:
        conditions = []
        
        for condition_key, condition_value in rule.conditions.items():
            if condition_key == 'min_mastery':
                if isinstance(condition_value, dict):
                    skill = condition_value.get('skill')
                    level = condition_value.get('level', 1)
                    min_mastery = condition_value.get('min_mastery', 0.5)
                    mastery = mastery_matrix.get((skill, level), 0.0)
                    normalized = 1.0 if mastery >= min_mastery else mastery / min_mastery
                    conditions.append(self.normalize_signal(normalized, 'mastery'))
                
            elif condition_key == 'skill_gap':
                if isinstance(condition_value, dict):
                    skill = condition_value.get('skill')
                    level = condition_value.get('level', 1)
                    max_mastery = condition_value.get('max_mastery', 0.5)
                    mastery = mastery_matrix.get((skill, level), 0.0)
                    normalized = 1.0 - (mastery / max_mastery) if mastery < max_mastery else 0.0
                    conditions.append(self.normalize_signal(normalized, 'mastery'))
                
            elif condition_key == 'activity_type':
                if isinstance(condition_value, dict):
                    activity_type = condition_value.get('type')
                    min_count = condition_value.get('min_count', 5)
                    count = activity_stats.get(f'{activity_type}_count', 0)
                    normalized = min(1.0, count / min_count) if min_count > 0 else 0.0
                    conditions.append(self.normalize_signal(normalized, 'activity_count'))
                
            elif condition_key == 'completed_modules':
                if isinstance(condition_value, dict):
                    required_modules = condition_value.get('modules', [])
                    completed = sum(1 for m in required_modules if m in learner.completed_modules)
                    normalized = completed / len(required_modules) if required_modules else 0.0
                    conditions.append(self.normalize_signal(normalized, 'score'))
                
            elif condition_key == 'level_progression':
                if isinstance(condition_value, dict):
                    skill = condition_value.get('skill')
                    current_level = condition_value.get('current_level', 1)
                    current_mastery = mastery_matrix.get((skill, current_level), 0.0)
                    conditions.append(self.normalize_signal(current_mastery, 'mastery'))
        
        return conditions
    
    def compute_rule_activation(self, conditions: List[float]) -> float:
        if not conditions:
            return 0.0
        return np.prod(conditions)
    
    def compute_rule_score(self, rule: Rule, activation: float) -> float:
        return activation * rule.weight


class ModuleScorer:    
    def __init__(self, modules: List[Module]):
        self.modules = {m.module_id: m for m in modules}
        self.module_prerequisites = {m.module_id: m.prerequisites for m in modules}
    
    def check_prerequisites(self, module_id: str, 
                           mastery_matrix: Dict[Tuple[str, int], float]) -> bool:
        if module_id not in self.modules:
            return False
        
        module = self.modules[module_id]
        
        for prereq in module.prerequisites:
            if isinstance(prereq, tuple) and len(prereq) == 2:
                skill_code, min_level = prereq
                mastery = mastery_matrix.get((skill_code, min_level), 0.0)
                if mastery < 0.5: 
                    return False
            elif isinstance(prereq, str):
                pass
        
        return True
    
    def check_time_feasibility(self, module_id: str, learner: Learner) -> bool:
        if module_id not in self.modules:
            return False
        
        module = self.modules[module_id]
        return learner.available_time_hours >= module.estimated_hours
    
    def aggregate_rule_scores(self, module_id: str, rule_scores: Dict[str, float],
                              rule_to_modules: Dict[str, List[str]]) -> float:
        total_score = 0.0
        
        for rule_id, score in rule_scores.items():
            recommended_modules = rule_to_modules.get(rule_id, [])
            if module_id in recommended_modules:
                total_score += score
        
        return total_score


class RecommendationEngine:
    
    def __init__(self, modules: List[Module], rules: List[Rule],
                 lambda_decay: float = 0.1):
        self.mastery_calculator = MasteryCalculator(lambda_decay=lambda_decay)
        self.rule_engine = RuleEngine()
        self.module_scorer = ModuleScorer(modules)
        
        for rule in rules:
            self.rule_engine.add_rule(rule)
        
        self.rule_to_modules: Dict[str, List[str]] = {}
        for rule in rules:
            self.rule_to_modules[rule.rule_id] = rule.recommended_modules
    
    def compute_mastery_matrix(self, learner: Learner) -> Dict[Tuple[str, int], float]:
        mastery_matrix = {}
        
        skill_levels = set()
        for attempt in learner.assessment_attempts:
            skill_levels.add((attempt.skill_code, attempt.level))
        
        for skill_code, level in skill_levels:
            mastery = self.mastery_calculator.compute_mastery_confidence(
                skill_code, level, learner.assessment_attempts
            )
            mastery_matrix[(skill_code, level)] = mastery
        
        return mastery_matrix
    
    def compute_activity_stats(self, learner: Learner) -> Dict[str, float]:
        stats = defaultdict(float)
        activity_types = set()
        
        for activity in learner.learning_activities:
            activity_types.add(activity.activity_type)
            stats[f'{activity.activity_type}_count'] += 1
            stats[f'{activity.activity_type}_time'] += activity.duration_minutes / 60.0
            stats['total_time'] += activity.duration_minutes / 60.0
        
        return dict(stats)
    
    def evaluate_rules(self, learner: Learner, mastery_matrix: Dict[Tuple[str, int], float],
                       activity_stats: Dict[str, float]) -> Dict[str, float]:
        rule_scores = {}
        
        for rule in self.rule_engine.rules:
            conditions = self.rule_engine.extract_rule_conditions(
                rule, learner, mastery_matrix, activity_stats
            )
            
            activation = self.rule_engine.compute_rule_activation(conditions)
            
            score = self.rule_engine.compute_rule_score(rule, activation)
            rule_scores[rule.rule_id] = score
        
        return rule_scores
    
    def score_modules(self, rule_scores: Dict[str, float]) -> Dict[str, float]:
        module_scores = defaultdict(float)
        
        for rule_id, score in rule_scores.items():
            recommended_modules = self.rule_to_modules.get(rule_id, [])
            for module_id in recommended_modules:
                module_scores[module_id] += score
        
        return dict(module_scores)
    
    def apply_constraints(self, module_scores: Dict[str, float], learner: Learner,
                         mastery_matrix: Dict[Tuple[str, int], float]) -> Dict[str, float]:
        filtered_scores = {}
        
        for module_id, score in module_scores.items():
            if not self.module_scorer.check_prerequisites(module_id, mastery_matrix):
                continue
            
            if not self.module_scorer.check_time_feasibility(module_id, learner):
                continue
            
            filtered_scores[module_id] = score
        
        return filtered_scores
    
    def generate_explanation(self, module_id: str, rule_scores: Dict[str, float],
                            mastery_matrix: Dict[Tuple[str, int], float],
                            contributing_rules: List[str]) -> Dict:
        explanation = {
            'module_id': module_id,
            'contributing_rules': [],
            'key_signals': {}
        }
        
        for rule_id in contributing_rules:
            rule = next((r for r in self.rule_engine.rules if r.rule_id == rule_id), None)
            if rule:
                explanation['contributing_rules'].append({
                    'rule_id': rule_id,
                    'rule_name': rule.name,
                    'rule_description': rule.description,
                    'score': rule_scores.get(rule_id, 0.0)
                })
        
        skill_masteries = [(k, v) for k, v in mastery_matrix.items() if v > 0.3]
        skill_masteries.sort(key=lambda x: x[1], reverse=True)
        explanation['key_signals'] = {
            f'{skill}_{level}': round(mastery, 3) 
            for (skill, level), mastery in skill_masteries[:5]
        }
        
        return explanation
    
    def recommend(self, learner: Learner, top_k: int = 10) -> List[Dict]:
        mastery_matrix = self.compute_mastery_matrix(learner)
        activity_stats = self.compute_activity_stats(learner)
        rule_scores = self.evaluate_rules(learner, mastery_matrix, activity_stats)
        module_scores = self.score_modules(rule_scores)
        filtered_scores = self.apply_constraints(module_scores, learner, mastery_matrix)
        sorted_modules = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations = []
        for module_id, score in sorted_modules[:top_k]:
            contributing_rules = [
                rule_id for rule_id, modules in self.rule_to_modules.items()
                if module_id in modules and rule_scores.get(rule_id, 0) > 0
            ]
            
            explanation = self.generate_explanation(
                module_id, rule_scores, mastery_matrix, contributing_rules
            )
            
            recommendations.append({
                'module_id': module_id,
                'module_name': self.module_scorer.modules[module_id].module_name,
                'score': round(score, 4),
                'explanation': explanation
            })
        
        return recommendations

