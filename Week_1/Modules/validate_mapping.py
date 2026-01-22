import csv
from collections import defaultdict

def validate_mapping():
    modules_path = 'modules.csv'
    mapping_path = 'module_sfia_mapping.csv'
    modules = {}
    with open(modules_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            modules[row['module_id']] = {
                'sfia_level': int(row['sfia_level']),
                'tag': row['tag'],
                'name': row['module_name']
            }
    mappings = defaultdict(list)
    with open(mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings[row['module_id']].append({
                'sfia_skill_code': row['sfia_skill_code'],
                'required_level': int(row['required_level']),
                'achieved_level': int(row['achieved_level'])
            })
    
    errors = []
    warnings = []
    
    for module_id, module_mappings in mappings.items():
        if module_id not in modules:
            errors.append(f"Module {module_id} not found in modules.csv")
            continue
        
        module = modules[module_id]
        module_level = module['sfia_level']
        num_skills = len(module_mappings)
        
        if num_skills < 3:
            errors.append(f"{module_id}: Only {num_skills} skills (minimum 3 required)")
        elif num_skills > 8:
            errors.append(f"{module_id}: {num_skills} skills (maximum 8 allowed)")
        
        for mapping in module_mappings:
            req_level = mapping['required_level']
            ach_level = mapping['achieved_level']
            
            if req_level > module_level:
                errors.append(f"{module_id} -> {mapping['sfia_skill_code']}: required_level ({req_level}) > module_level ({module_level})")
            
            if ach_level < module_level:
                errors.append(f"{module_id} -> {mapping['sfia_skill_code']}: achieved_level ({ach_level}) < module_level ({module_level})")
            
            if ach_level > 7:
                errors.append(f"{module_id} -> {mapping['sfia_skill_code']}: achieved_level ({ach_level}) > 7")
            
            if req_level >= ach_level:
                warnings.append(f"{module_id} -> {mapping['sfia_skill_code']}: required_level ({req_level}) >= achieved_level ({ach_level})")
    
    print("Validation Results:")
    print(f"Total modules: {len(mappings)}")
    print(f"Total mappings: {sum(len(m) for m in mappings.values())}")
    print()
    
    if errors:
        print(f"ERRORS ({len(errors)}):")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
        print()
    else:
        print("No errors found!")
        print()
    
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
        print()
    else:
        print("No warnings!")
        print()
    
    skill_counts = [len(m) for m in mappings.values()]
    print("Statistics:")
    print(f" Skills per module - Min: {min(skill_counts)}, Max: {max(skill_counts)}, Avg: {sum(skill_counts)/len(skill_counts):.2f}")
    
    return len(errors) == 0

if __name__ == '__main__':
    success = validate_mapping()
    exit(0 if success else 1)