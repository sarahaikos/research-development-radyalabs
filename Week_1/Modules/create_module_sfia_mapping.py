import csv
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set

def parse_sfia_tags(tag_string: str) -> Set[str]:
    if not tag_string or tag_string.strip() == '':
        return set()
    tags = [t.strip() for t in tag_string.replace('"', '').split(',')]
    return set(tags)

def load_modules(csv_path: str) -> List[Dict]:
    modules = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            modules.append({
                'module_id': row['module_id'],
                'module_name': row['module_name'],
                'tag': row['tag'],
                'sfia_level': int(row['sfia_level'])
            })
    return modules

def load_sfia_skills(csv_path: str) -> List[Dict]:
    skills = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['Code'] or row['Code'].strip() == '':
                continue
            tags = parse_sfia_tags(row.get('tags', ''))
            skills.append({
                'code': row['Code'],
                'category': row['Category'],
                'subcategory': row['Subcategory'],
                'skill': row['Skill'],
                'tags': tags
            })
    return skills

def get_relevant_skills(module: Dict, all_skills: List[Dict]) -> List[Dict]:
    module_tag = module['tag']
    module_name = module['module_name'].lower()
    matching_skills = []
    for skill in all_skills:
        if module_tag in skill['tags']:
            matching_skills.append(skill)    
    return matching_skills

def select_skills_for_module(module: Dict, relevant_skills: List[Dict]) -> List[Dict]:
    module_name = module['module_name'].lower()
    module_level = module['sfia_level']
    module_tag = module['tag']
    module_skill_mappings = {
        # DATA modules
        'data literacy': ['DATM', 'DAAN', 'VISL'],
        'programming for data': ['PROG', 'DAAN', 'DATS'],
        'statistics': ['DAAN', 'DATS', 'DTAN'],
        'data types': ['DATM', 'DTAN', 'DBDS'],
        'spreadsheet': ['DAAN', 'DATM'],
        'python or r': ['PROG', 'DAAN', 'DATS', 'DENG'],
        'sql': ['DBDS', 'DBAD', 'DAAN', 'DATM'],
        'data cleaning': ['DENG', 'DATM', 'DAAN'],
        'exploratory': ['DAAN', 'DATS', 'VISL', 'DTAN'],
        'machine learning': ['MLNG', 'DATS', 'DAAN', 'DENG'],
        'feature engineering': ['MLNG', 'DENG', 'DTAN', 'DAAN'],
        'data engineering': ['DENG', 'DATM', 'DBAD', 'SINT'],
        'visualization': ['VISL', 'DAAN', 'BINT'],
        'model evaluation': ['MLNG', 'DATS', 'QUAS'],
        'scalable pipelines': ['DENG', 'SINT', 'ARCH', 'ITOP'],
        'advanced ml': ['MLNG', 'DATS', 'ARCH'],
        'model deployment': ['MLNG', 'DEPL', 'SLEN'],
        'data governance': ['DATM', 'GOVN', 'AIDE', 'PEDP'],
        'analytics for business': ['BINT', 'DAAN', 'BUSA'],
        'data platform': ['ARCH', 'DENG', 'IFDN', 'ITOP'],
        'ai strategy': ['ARCH', 'ITSP', 'AIDE', 'INOV'],
        'large scale data': ['DENG', 'ARCH', 'ITOP'],
        'responsible ai': ['AIDE', 'GOVN', 'BURM'],
        'enterprise data': ['ARCH', 'STPL', 'GOVN', 'IRMG'],
        'data transformation': ['ARCH', 'OCDV', 'CIPM'],
        'ai governance': ['GOVN', 'AIDE', 'STPL'],
        
        # DEVO modules
        'devops concepts': ['DEPL', 'RELM', 'ITOP'],
        'version control': ['CFMG', 'PROG', 'SINT'],
        'command line': ['ITOP', 'SYSP'],
        'cicd': ['DEPL', 'RELM', 'SINT', 'SLEN'],
        'infrastructure': ['IFDN', 'ITOP', 'ITMG'],
        'container': ['DEPL', 'SINT', 'ITOP'],
        'monitoring': ['ITOP', 'AVMT', 'PBMG'],
        'infrastructure as code': ['IFDN', 'DEPL', 'ARCH'],
        'container orchestration': ['DEPL', 'SINT', 'ARCH'],
        'devops security': ['SCTY', 'DEPL', 'VUAS'],
        'advanced cicd': ['DEPL', 'RELM', 'SLEN', 'ARCH'],
        'cloud native': ['ARCH', 'DEPL', 'IFDN'],
        'reliability': ['AVMT', 'ITOP', 'ARCH'],
        'devsecops': ['SCTY', 'DEPL', 'VUAS', 'SCAD'],
        'devops architecture': ['ARCH', 'IFDN', 'STPL'],
        'platform engineering': ['ARCH', 'ITOP', 'SLEN'],
        'large scale reliability': ['ARCH', 'AVMT', 'STPL'],
        'devops transformation': ['STPL', 'OCDV', 'CIPM'],
        'devops governance': ['GOVN', 'ARCH', 'MEAS'],
        
        # SOFT modules
        'programming fundamentals': ['PROG', 'SWDN'],
        'algorithms': ['PROG', 'SWDN', 'DESN'],
        'software development': ['PROG', 'SWDN', 'SLEN'],
        'object oriented': ['PROG', 'SWDN', 'DESN'],
        'data structures': ['PROG', 'SWDN', 'DESN'],
        'software testing': ['TEST', 'NFTS', 'QUAS'],
        'version control': ['CFMG', 'SINT', 'PROG'],
        'software design': ['SWDN', 'DESN', 'ARCH'],
        'api design': ['SWDN', 'DESN', 'SINT'],
        'secure coding': ['PROG', 'SCTY', 'SWDN'],
        'system design': ['DESN', 'ARCH', 'IFDN'],
        'software architecture': ['ARCH', 'SWDN', 'STPL'],
        'performance': ['SWDN', 'ARCH', 'MEAS'],
        'quality engineering': ['QUAS', 'TEST', 'QUMG'],
        'technical leadership': ['ARCH', 'RLMT', 'PEMT'],
        'large scale system': ['ARCH', 'STPL', 'IFDN'],
        'software modernization': ['ARCH', 'STPL', 'INOV'],
        'engineering best practices': ['GOVN', 'QUAS', 'METL'],
        'enterprise software': ['STPL', 'ARCH', 'ITSP'],
        'technology vision': ['ITSP', 'INOV', 'STPL'],
        
        # CYBR modules
        'cybersecurity fundamentals': ['SCTY', 'INAS'],
        'threat awareness': ['THIN', 'SCTY'],
        'security hygiene': ['SCTY', 'INAS'],
        'network security': ['SCTY', 'NTDS', 'INAS'],
        'identity and access': ['IAMT', 'SCTY'],
        'vulnerability assessment': ['VUAS', 'SCTY', 'INAS'],
        'security architecture': ['ARCH', 'SCTY', 'INAS'],
        'incident detection': ['SCAD', 'USUP', 'SCTY'],
        'risk assessment': ['BURM', 'SCTY', 'INAS'],
        'security operations': ['SCAD', 'SCTY', 'USUP'],
        'penetration testing': ['PENT', 'VUAS', 'SCAD'],
        'governance risk': ['GOVN', 'BURM', 'AUDT'],
        'enterprise security': ['ARCH', 'STPL', 'SCTY'],
        'cyber risk strategy': ['BURM', 'ITSP', 'SCTY'],
        'security program': ['GOVN', 'SCAD', 'STPL'],
        'cybersecurity strategy': ['STPL', 'ITSP', 'GOVN'],
        'security policy': ['GOVN', 'STPL', 'PEDP'],
        
        # CLOU modules
        'cloud computing': ['IFDN', 'ITMG', 'ARCH'],
        'cloud service': ['ITMG', 'IFDN'],
        'cloud resource': ['ITOP', 'ITMG', 'CPMG'],
        'cloud networking': ['NTDS', 'IFDN'],
        'cloud security': ['SCTY', 'INAS', 'IFDN'],
        'cloud architecture': ['ARCH', 'IFDN', 'STPL'],
        'cost optimization': ['COMG', 'MEAS', 'ITOP'],
        'multi cloud': ['ARCH', 'IFDN', 'STPL'],
        'cloud migration': ['ARCH', 'DEPL', 'STPL'],
        'cloud governance': ['GOVN', 'ITMG', 'ARCH'],
        'cloud native': ['ARCH', 'SWDN', 'DEPL'],
        'enterprise cloud': ['STPL', 'ARCH', 'GOVN'],
        'cloud operating': ['STPL', 'ITMG', 'ARCH'],
        'digital transformation': ['STPL', 'ITSP', 'OCDV'],
        'cloud adoption': ['STPL', 'CIPM', 'OCDV'],
    }
    scored_skills = []
    matched_skills = []
    for key, skill_codes in module_skill_mappings.items():
        if key in module_name:
            for skill in relevant_skills:
                if skill['code'] in skill_codes:
                    matched_skills.append(skill)
    
    if matched_skills:
        seen = set()
        unique_matched = []
        for skill in matched_skills:
            if skill['code'] not in seen:
                seen.add(skill['code'])
                unique_matched.append(skill)
        matched_skills = unique_matched
    
    for skill in relevant_skills:
        score = 0
        skill_name = skill['skill'].lower()
        skill_code = skill['code']
        category = skill['category'].lower()
        subcategory = skill['subcategory'].lower()
        
        if skill in matched_skills:
            score += 50
        
        module_keywords = set(module_name.split())
        skill_keywords = set(skill_name.split())
        common_keywords = module_keywords.intersection(skill_keywords)
        score += len(common_keywords) * 5
        
        if module_tag == 'DATA':
            if 'data' in category or 'data' in subcategory:
                score += 10
            if skill_code in ['DAAN', 'DATS', 'DATM', 'DENG', 'MLNG', 'DTAN', 'DBDS', 'VISL', 'BINT', 'AIDE']:
                score += 15
        elif module_tag == 'DEVO':
            if 'technology' in category or 'deployment' in subcategory:
                score += 10
            if skill_code in ['DEPL', 'RELM', 'CFMG', 'SINT', 'SLEN', 'ITOP']:
                score += 15
        elif module_tag == 'SOFT':
            if 'development' in category or 'systems development' in subcategory:
                score += 10
            if skill_code in ['PROG', 'SWDN', 'TEST', 'DESN', 'SINT', 'SLEN']:
                score += 15
        elif module_tag == 'CYBR':
            if 'security' in category or 'security' in subcategory:
                score += 10
            if skill_code in ['SCTY', 'VUAS', 'PENT', 'SCAD', 'IAMT', 'INAS', 'THIN']:
                score += 15
        elif module_tag == 'CLOU':
            if 'infrastructure' in category or 'infrastructure' in subcategory:
                score += 10
            if skill_code in ['IFDN', 'ARCH', 'ITOP', 'ITMG', 'DEPL', 'NTDS']:
                score += 15
        scored_skills.append((score, skill))
    
    scored_skills.sort(key=lambda x: x[0], reverse=True)
    
    if module_level <= 2:
        target_min, target_max = 3, 5
    elif module_level <= 4:
        target_min, target_max = 4, 6
    else:
        target_min, target_max = 5, 8
    
    selected = []
    seen_codes = set()
    top_score = scored_skills[0][0] if scored_skills else 0
    threshold = max(5, top_score * 0.3)
    
    for score, skill in scored_skills:
        if len(selected) >= target_max:
            break
        if skill['code'] not in seen_codes:
            if module_level <= 3 and score < threshold and len(selected) >= target_min:
                break
            selected.append(skill)
            seen_codes.add(skill['code'])
    
    if len(selected) < target_min:
        for score, skill in scored_skills:
            if len(selected) >= target_max:
                break
            if skill['code'] not in seen_codes:
                selected.append(skill)
                seen_codes.add(skill['code'])
    
    return selected[:8] if len(selected) >= 3 else (selected if len(selected) >= 3 else relevant_skills[:3])

def determine_levels(module_level: int) -> Tuple[int, int]:
    if module_level == 2:
        required_level = 1
        achieved_level = 2
    elif module_level == 7:
        required_level = 6
        achieved_level = 7
    else:
        required_level = module_level - 1
        achieved_level = module_level  
    return required_level, achieved_level

def create_mappings(modules: List[Dict], skills: List[Dict]) -> List[Dict]:
    mappings = []
    for module in modules:
        relevant_skills = get_relevant_skills(module, skills)
        if len(relevant_skills) == 0:
            print(f"Warning: No SFIA skills found for module {module['module_id']} with tag {module['tag']}")
            continue
        selected_skills = select_skills_for_module(module, relevant_skills)
        required_level, achieved_level = determine_levels(module['sfia_level'])
        for skill in selected_skills:
            mappings.append({
                'module_id': module['module_id'],
                'sfia_skill_code': skill['code'],
                'required_level': required_level,
                'achieved_level': achieved_level
            })
    return mappings

def main():
    modules_path = 'modules.csv'
    sfia_path = '../SFIA/sfia_code_category_tag_summary.csv'
    output_path = 'module_sfia_mapping.csv'
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['module_id', 'sfia_skill_code', 'required_level', 'achieved_level']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mappings)
    
    module_counts = defaultdict(int)
    for mapping in mappings:
        module_counts[mapping['module_id']] += 1
    
    print("Summary:")
    print(f"Total modules mapped: {len(module_counts)}")
    print(f"Average skills per module: {len(mappings) / len(module_counts):.2f}")
    print(f"Min skills per module: {min(module_counts.values())}")
    print(f"Max skills per module: {max(module_counts.values())}")

if __name__ == '__main__':
    main()