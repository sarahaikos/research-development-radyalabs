import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import defaultdict
from pathlib import Path

TAG_COLORS = {
    'SOFTWARE': '#4A90A4',
    'DATA': '#2171B5',
    'UIUX': '#6BAED6',
    'CLOUD': '#9ECAE1',
    'INFRA': '#9ECAE1',
    'EMBEDDED': '#4292C6',
    'MANAGEMENT': '#084594',
    'SAFETY': '#6BAED6',
    'CONTENT': '#8BC4D8',
    'SCIENCE': '#5B9DB0',
    'AI': '#2171B5',
    'QA': '#4A90A4',
    'ARCH': '#3A7A8C',
    'NETWORK': '#9ECAE1',
    'HARDWARE': '#4292C6',
    'DEVOPS': '#4A90A4',
    'MEDIA': '#6BAED6',
}

LEVEL_COLORS = {
    1: '#2C3E50',  # Dark blue-gray for root
    2: '#34495E',  # Medium gray for subcategory
    3: '#7F8C8D',  # Light gray for sub-area
    4: '#ECF0F1',  # Very light gray for skills
}

SUBCATEGORY_COLORS = {
    'Systems development': '#5B9BD5',      # Calm blue
    'Data and analytics': '#70AD47',       # Calm green
    'User centred design': '#9E7BB5',     # Calm purple
    'Content management': '#4BACC6',      # Calm teal
    'Computational science': '#F4B084',   # Calm peach
    'default': '#95A5A6'                   # Default light gray
}

def extract_all_tags(tags_str):
    if pd.isna(tags_str) or not isinstance(tags_str, str):
        return []
    tags = [t.strip().upper() for t in tags_str.split(',')]
    return tags

def load_and_structure_data_by_tags(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df['Code'].notna() & (df['Code'].astype(str).str.strip() != '')]
    hierarchy = defaultdict(list)
    
    for _, row in df.iterrows():
        skill = str(row.get('Skill', '')).strip()
        code = str(row.get('Code', '')).strip()
        tags_str = row.get('Tags', '')
        subcategory = str(row.get('Subcategory', '')).strip()
        sub_area = str(row.get('Sub-area', '')).strip()
        
        if code:
            skill_label = f"{code} – {skill}" if skill else code
            tags = extract_all_tags(tags_str)
            
            if not tags:
                tags = ['UNCATEGORIZED']
            
            for tag in tags:
                hierarchy[tag].append({
                    'code': code,
                    'skill': skill,
                    'label': skill_label,
                    'subcategory': subcategory,
                    'sub_area': sub_area,
                    'all_tags': tags
                })
    
    for tag in hierarchy:
        hierarchy[tag].sort(key=lambda x: x['code'])
    
    return hierarchy

def calculate_tag_layout(hierarchy, fig_height_inches=20):
    layout = {
        'tags': {},
        'skills': {},
        'subcategories': {}
    }
    
    y_spacing = 0.22  
    tag_spacing = 0.6  
    
    x_levels = [2.0, 5.0, 8.0]
    
    subcategory_positions = {}
    subcategory_y_positions = {}
    
    current_y = 0.1
    y_positions = []
    
    tag_priority = ['SOFTWARE', 'DATA', 'UIUX', 'CLOUD', 'INFRA', 'EMBEDDED', 
                    'MANAGEMENT', 'SAFETY', 'CONTENT', 'SCIENCE', 'AI', 'QA',
                    'ARCH', 'NETWORK', 'HARDWARE', 'DEVOPS', 'MEDIA']
    
    def tag_sort_key(tag):
        if tag in tag_priority:
            return (0, tag_priority.index(tag))
        return (1, tag)
    
    sorted_tags = sorted(hierarchy.keys(), key=tag_sort_key)
    
    for tag in sorted_tags:
        skills = hierarchy[tag]
        tag_y_start = current_y
        
        tag_height = len(skills) * y_spacing + 0.2
        
        tag_y = tag_y_start + tag_height / 2
        layout['tags'][tag] = {
            'x': x_levels[1],
            'y': tag_y,
            'height': tag_height
        }
        
        skill_y = tag_y_start + 0.1
        for skill_data in skills:
            layout['skills'][(tag, skill_data['code'])] = {
                'x': x_levels[2],
                'y': skill_y,
                'subcategory': skill_data['subcategory']
            }
            
            subcategory = skill_data['subcategory']
            if subcategory:
                if subcategory not in subcategory_y_positions:
                    subcategory_y_positions[subcategory] = []
                subcategory_y_positions[subcategory].append(skill_y)
            
            skill_y += y_spacing
        
        current_y += tag_height + tag_spacing
        y_positions.append(current_y)
    
    subcategory_avg_y = {
        subcat: sum(y_list) / len(y_list)
        for subcat, y_list in subcategory_y_positions.items()
    }

    sorted_subcats = sorted(subcategory_avg_y.items(), key=lambda x: x[1])

    SUBCAT_GAP = 3
    SUBCAT_START_Y = min(subcategory_avg_y.values())

    for i, (subcategory, _) in enumerate(sorted_subcats):
        layout['subcategories'][subcategory] = {
            'x': x_levels[2],
            'y': SUBCAT_START_Y + i * SUBCAT_GAP
        }

    max_y = max(y_positions) if y_positions else 1.0
    scale_factor = 0.9 / max_y if max_y > 0 else 1.0
    offset = 0.05
    
    def normalize_y(y):
        return 1.0 - (y * scale_factor + offset)
    
    for key in layout['tags']:
        layout['tags'][key]['y'] = normalize_y(layout['tags'][key]['y'])
    for key in layout['skills']:
        layout['skills'][key]['y'] = normalize_y(layout['skills'][key]['y'])
    for key in layout['subcategories']:
        layout['subcategories'][key]['y'] = normalize_y(layout['subcategories'][key]['y'])
    
    return layout, x_levels

def create_tag_based_diagram(hierarchy, output_path):
    total_skills = sum(len(skills) for skills in hierarchy.values())
    fig_height = max(18, total_skills * 0.35)
    fig, ax = plt.subplots(figsize=(20, fig_height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    layout, x_levels = calculate_tag_layout(hierarchy, fig_height)
    
    x_scale = 0.9
    x_offset = 0.04
    x_levels_norm = [(x * x_scale / 10.0) + x_offset for x in x_levels]
    
    max_subcat_x = 0.90
    if x_levels_norm[2] > max_subcat_x:
        x_levels_norm[2] = max_subcat_x
    
    # level 2
    for tag, pos in layout['tags'].items():
        tag_x = x_levels_norm[0]
        tag_y = pos['y']
        tag_color = TAG_COLORS.get(tag, '#BDC3C7')
        
        tag_box = FancyBboxPatch(
            (tag_x - 0.04, tag_y - 0.012),
            0.08, 0.024,
            boxstyle="round,pad=0.003",
            edgecolor='black',
            facecolor=tag_color,
            linewidth=1.5,
            zorder=9
        )
        ax.add_patch(tag_box)
        ax.text(tag_x, tag_y, tag,
                ha='center', va='center', fontsize=10, fontweight='bold', color='white',
                zorder=10)
        
        # level 3
        for (tag2, code), skill_pos in layout['skills'].items():
            if tag2 != tag:
                continue
            
            skill_x = x_levels_norm[1]
            skill_y = skill_pos['y']
            
            skill_data = None
            for skill in hierarchy[tag]:
                if skill['code'] == code:
                    skill_data = skill
                    break
            
            if not skill_data:
                continue
            
            subcategory = skill_pos.get('subcategory', '')
            indicator_color = SUBCATEGORY_COLORS.get(subcategory, SUBCATEGORY_COLORS['default']) if subcategory else '#BDC3C7'
            
            tag_indicator = mpatches.Rectangle(
                (skill_x - 0.02, skill_y - 0.005),
                0.005, 0.01,
                facecolor=indicator_color,
                edgecolor='none',
                zorder=8
            )
            ax.add_patch(tag_indicator)
            
            label = skill_data['label']
            max_chars = 50
            if len(label) > max_chars:
                label = label[:max_chars-3] + '...'
            
            ax.text(skill_x - 0.015, skill_y, label,
                    ha='left', va='center', fontsize=7, color='black',
                    zorder=8)
            
            arrow = FancyArrowPatch(
                (tag_x + 0.04, tag_y),
                (skill_x - 0.02, skill_y),
                arrowstyle='->', lw=1, color='#D5DBDB', alpha=0.5, zorder=1
            )
            ax.add_patch(arrow)
            
            # level 4 - connection from skill to subcategory (diagonal like tag-to-skill)
            subcategory = skill_pos.get('subcategory', '')
            if subcategory and subcategory in layout['subcategories']:
                subcat_x = x_levels_norm[2]
                subcat_y = layout['subcategories'][subcategory]['y']
                
                # Use a clear, fixed starting point - from the right edge of skill text area
                # This matches the style: tag connection ends at skill_x - 0.02, so start from there
                connection_start_x = skill_x - 0.02  # Clear starting point, aligned with tag-to-skill connection end
                
                # Diagonal line from skill to subcategory box (like tag-to-skill connection)
                # End point connects directly to the left edge of subcategory box
                arrow = FancyArrowPatch(
                    (connection_start_x, skill_y),
                    (subcat_x - 0.04, subcat_y),  # Connect directly to left edge of subcategory box
                    arrowstyle='->',
                    lw=1.0,
                    color='#E8E8E8',  # Softer light gray color
                    alpha=0.5,  # More transparent
                    zorder=1  # Lower zorder so it's behind text
                )
                ax.add_patch(arrow)
    
    for subcategory, pos in layout['subcategories'].items():
        subcat_x = x_levels_norm[2]
        subcat_y = pos['y']
        
        subcat_color = SUBCATEGORY_COLORS.get(subcategory, SUBCATEGORY_COLORS['default'])
        
        subcat_box = FancyBboxPatch(
            (subcat_x - 0.04, subcat_y - 0.012),
            0.08, 0.024,
            boxstyle="round,pad=0.002",
            edgecolor='black',
            facecolor=subcat_color,
            linewidth=1.2,
            zorder=6
        )
        ax.add_patch(subcat_box)
        ax.text(subcat_x, subcat_y, subcategory,
                ha='center', va='center', fontsize=8.5, fontweight='bold', color='white',
                zorder=7)

    fig.suptitle('SFIA Skills Mapping for the Development and Implementation',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.5)
    print(f"Hierarchy diagram saved to: {output_path}")
    plt.close()

def create_table_mapping_by_tags(hierarchy, output_path):
    rows = []
    
    for tag in sorted(hierarchy.keys()):
        for skill_data in hierarchy[tag]:
            rows.append({
                'Tag': tag,
                'Code': skill_data['code'],
                'Skill': skill_data['skill'],
                'Subcategory': skill_data['subcategory'],
                'Sub-area': skill_data['sub_area']
            })
    
    df_table = pd.DataFrame(rows)
    df_table.to_csv(output_path, index=False)
    print(f"Table mapping saved to: {output_path}")
    return df_table

def main():
    base_path = Path(__file__).parent
    csv_path = base_path / "SFIA_Development_and_Implementation.csv"
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    hierarchy = load_and_structure_data_by_tags(csv_path)
    
    diagram_path = base_path / "SFIA_Development_Implementation_Hierarchy.png"
    create_tag_based_diagram(hierarchy, diagram_path)
    
    table_path = base_path / "SFIA_Development_Implementation_Mapping.csv"
    create_table_mapping_by_tags(hierarchy, table_path)
    
    print("Summary:")
    print(f"  Tags: {len(hierarchy)}")
    total_skills = sum(len(skills) for skills in hierarchy.values())
    print(f"  Skills: {total_skills}")

if __name__ == "__main__":
    main()

