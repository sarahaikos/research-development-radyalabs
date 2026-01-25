import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
from collections import defaultdict

sys.path.append(str(Path(__file__).parent))

from module_recommendation_engine import RecommendationEngine, Learner
from data_loader import (
    load_modules_from_csv,
    load_module_mappings,
    populate_module_data,
    generate_default_rules,
    load_learner_from_csv
)


def main():
    base_path = Path(__file__).parent.parent
    modules_csv = base_path / 'Modules' / 'modules.csv'
    mappings_csv = base_path / 'Modules' / 'module_sfia_mapping.csv'
    
    model_path = Path(__file__).parent
    assessments_csv = model_path / 'sample_learner_assessments.csv'
    activities_csv = model_path / 'sample_learner_activities.csv'
    profile_csv = model_path / 'sample_learner_profile.csv'
    
    print("-" * 80)
    print("Rule Based Model")
    print("-" * 80)
    
    print("\n[1] Loading modules...")
    modules = load_modules_from_csv(str(modules_csv))
    print(f"    Loaded {len(modules)} modules")
    
    print("\n[2] Loading module-skill mappings...")
    mappings = load_module_mappings(str(mappings_csv))
    print(f"    Loaded mappings for {len(mappings)} modules")
    
    print("\n[3] Populating module data...")
    populate_module_data(modules, mappings)
    print("    Module data populated")
    
    print("\n[4] Generating recommendation rules...")
    rules = generate_default_rules(modules, mappings)
    print(f"    Generated {len(rules)} rules")
    
    print("\n[5] Initializing recommendation engine...")
    engine = RecommendationEngine(
        modules=modules,
        rules=rules,
        lambda_decay=0.1
    )
    print("    Engine initialized")
    
    print("\n[6] Loading learner data from CSV...")
    learner = load_learner_from_csv(
        learner_id='learner_001',
        assessments_csv=str(assessments_csv),
        activities_csv=str(activities_csv),
        profile_csv=str(profile_csv)
    )
    print(f"    Loaded learner with:")
    print(f"      - {len(learner.assessment_attempts)} assessment attempts")
    print(f"      - {len(learner.learning_activities)} learning activities")
    print(f"      - {learner.available_time_hours:.1f} hours available per week")
    print(f"      - {len(learner.completed_modules)} completed modules")
    
    print("\n[7] Generating recommendations...")
    recommendations = engine.recommend(learner, top_k=10)
    print(f"    Generated {len(recommendations)} recommendations")
    
    print("\n" + "-" * 80)
    print("TOP RECOMMENDATIONS")
    print("-" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n[{i}] {rec['module_name']} ({rec['module_id']})")
        print(f"    Score: {rec['score']:.4f}")
        print(f"    Contributing Rules:")
        for rule_info in rec['explanation']['contributing_rules'][:3]:
            print(f"      - {rule_info['rule_name']} (score: {rule_info['score']:.4f})")
        print(f"    Key Signals:")
        for signal, value in list(rec['explanation']['key_signals'].items())[:3]:
            print(f"      - {signal}: {value:.3f}")
    
    print("\n" + "-" * 80)
    print("LEARNER MASTERY SUMMARY")
    print("-" * 80)
    
    mastery_matrix = engine.compute_mastery_matrix(learner)
    skill_summary = {}
    
    for (skill, level), mastery in mastery_matrix.items():
        if skill not in skill_summary:
            skill_summary[skill] = {}
        skill_summary[skill][level] = mastery
    
    for skill, levels in sorted(skill_summary.items())[:10]:
        print(f"\n{skill}:")
        for level in sorted(levels.keys()):
            mastery = levels[level]
            bar = '█' * int(mastery * 20)
            print(f"  Level {level}: {bar} {mastery:.3f}")
    
    print("\n" + "-" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("-" * 80)
    
    create_visualizations(recommendations, mastery_matrix, engine, learner)
    
    print("\nVisualizations saved to 'recommendation_visualizations.png'")


def create_visualizations(recommendations, mastery_matrix, engine, learner):
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    base_color = '#4A90A4' 
    color_palette = [
        '#6BA3B8',
        '#4A90A4',
        '#3A7A8C', 
        '#5B9DB0', 
        '#4A90A4', 
        '#6BA3B8', 
        '#3A7A8C',
        '#5B9DB0', 
        '#4A90A4',  
        '#6BA3B8', 
    ]
    
    colors_heatmap = ['#E8F4F8', '#B8DCE8', '#8BC4D8', '#6BA3B8', '#4A90A4', '#3A7A8C']
    n_bins = 256
    cmap_custom = LinearSegmentedColormap.from_list('calm_blue', colors_heatmap, N=n_bins)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    if recommendations:
        top_n = min(10, len(recommendations))
        module_names = [rec['module_name'][:30] + '...' if len(rec['module_name']) > 30 
                       else rec['module_name'] for rec in recommendations[:top_n]]
        scores = [rec['score'] for rec in recommendations[:top_n]]
        colors = [color_palette[i % len(color_palette)] for i in range(len(scores))]
        
        bars = ax1.barh(range(len(module_names)), scores, color=colors)
        ax1.set_yticks(range(len(module_names)))
        ax1.set_yticklabels(module_names, fontsize=9)
        ax1.set_xlabel('Recommendation Score', fontsize=11, fontweight='bold')
        ax1.set_title('Top Recommended Modules', fontsize=13, fontweight='bold', pad=15)
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score + max(scores) * 0.01, i, f'{score:.3f}', 
                    va='center', fontsize=8)
    
    # heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    if mastery_matrix:
        skills = sorted(set(skill for skill, level in mastery_matrix.keys()))
        levels = sorted(set(level for skill, level in mastery_matrix.keys()))
        
        heatmap_data = np.zeros((len(skills), len(levels)))
        for i, skill in enumerate(skills):
            for j, level in enumerate(levels):
                heatmap_data[i, j] = mastery_matrix.get((skill, level), 0.0)
        
        # show top 15 skills by average mastery
        skill_avg_mastery = [(skill, np.mean([mastery_matrix.get((skill, level), 0) 
                                             for level in levels])) 
                            for skill in skills]
        skill_avg_mastery.sort(key=lambda x: x[1], reverse=True)
        top_skills = [s[0] for s in skill_avg_mastery[:15]]
        
        top_heatmap = np.array([[mastery_matrix.get((skill, level), 0.0) 
                                for level in levels] for skill in top_skills])
        
        im = ax2.imshow(top_heatmap, aspect='auto', cmap=cmap_custom, vmin=0, vmax=1)
        ax2.set_xticks(range(len(levels)))
        ax2.set_xticklabels([f'L{l}' for l in levels], fontsize=9)
        ax2.set_yticks(range(len(top_skills)))
        ax2.set_yticklabels(top_skills, fontsize=8)
        ax2.set_xlabel('SFIA Level', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Skill Code', fontsize=11, fontweight='bold')
        ax2.set_title('Skill Mastery Heatmap (Top 15 Skills)', fontsize=13, fontweight='bold', pad=15)
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Mastery Confidence', fontsize=10)
        
        for i in range(len(top_skills)):
            for j in range(len(levels)):
                text_color = "white" if top_heatmap[i, j] > 0.5 else "#2C5F6B"
                text = ax2.text(j, i, f'{top_heatmap[i, j]:.2f}',
                              ha="center", va="center", color=text_color, fontsize=7, fontweight='medium')
    
    # recommendation score distribution
    ax3 = fig.add_subplot(gs[1, 0])
    if recommendations:
        scores = [rec['score'] for rec in recommendations]
        tags = [rec['module_id'].split('_')[0] for rec in recommendations]
        
        tag_scores = defaultdict(list)
        for tag, score in zip(tags, scores):
            tag_scores[tag].append(score)
        
        tag_means = {tag: np.mean(scores) for tag, scores in tag_scores.items()}
        tag_counts = {tag: len(scores) for tag, scores in tag_scores.items()}
        
        tags_sorted = sorted(tag_means.keys(), key=lambda x: tag_means[x], reverse=True)
        means = [tag_means[t] for t in tags_sorted]
        counts = [tag_counts[t] for t in tags_sorted]
        
        x_pos = np.arange(len(tags_sorted))
        bar_colors = [color_palette[i % len(color_palette)] for i in range(len(tags_sorted))]
        bars = ax3.bar(x_pos, means, color=bar_colors)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(tags_sorted, fontsize=10)
        ax3.set_ylabel('Average Recommendation Score', fontsize=11, fontweight='bold')
        ax3.set_title('Recommendation Scores by Category', fontsize=13, fontweight='bold', pad=15)
        ax3.grid(axis='y', alpha=0.3)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'n={count}',
                    ha='center', va='bottom', fontsize=9)
    
    # mastery distribution by level
    ax4 = fig.add_subplot(gs[1, 1])
    if mastery_matrix:
        level_masteries = defaultdict(list)
        for (skill, level), mastery in mastery_matrix.items():
            level_masteries[level].append(mastery)
        
        levels_sorted = sorted(level_masteries.keys())
        data_to_plot = [level_masteries[level] for level in levels_sorted]
        labels = [f'Level {l}' for l in levels_sorted]
        
        bp = ax4.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
        box_colors = [color_palette[i % len(color_palette)] for i in range(len(bp['boxes']))]
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Mastery Confidence', fontsize=11, fontweight='bold')
        ax4.set_xlabel('SFIA Level', fontsize=11, fontweight='bold')
        ax4.set_title('Mastery Distribution by Level', fontsize=13, fontweight='bold', pad=15)
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, 1.1)
    
    # rule activation analysis
    ax5 = fig.add_subplot(gs[2, :])
    
    rule_scores = engine.evaluate_rules(
        learner, 
        engine.compute_mastery_matrix(learner),
        engine.compute_activity_stats(learner)
    )
    
    if rule_scores:
        sorted_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        rule_ids = [r[0][:40] + '...' if len(r[0]) > 40 else r[0] for r in sorted_rules]
        scores = [r[1] for r in sorted_rules]
        
        colors = [color_palette[i % len(color_palette)] for i in range(len(scores))]
        bars = ax5.barh(range(len(rule_ids)), scores, color=colors)
        ax5.set_yticks(range(len(rule_ids)))
        ax5.set_yticklabels(rule_ids, fontsize=8)
        ax5.set_xlabel('Rule Score', fontsize=11, fontweight='bold')
        ax5.set_title('Top 20 Rule Activations', fontsize=13, fontweight='bold', pad=15)
        ax5.invert_yaxis()
        ax5.grid(axis='x', alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0:
                ax5.text(score + max(scores) * 0.01, i, f'{score:.3f}', 
                        va='center', fontsize=7)
    
    fig.suptitle('Module Recommendation Engine - Analysis Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('recommendation_visualizations.png', dpi=150, bbox_inches='tight')
    print("Visualizations saved successfully!")

if __name__ == '__main__':
    main()