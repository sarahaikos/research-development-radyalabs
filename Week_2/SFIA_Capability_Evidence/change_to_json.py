import pandas as pd
import json

full_df = pd.read_csv('SFIA_Capabilities_Full.csv')

sfia_json = {}
for code, group in full_df.groupby('SFIA_Code'):
    sfia_json[code] = {
        "skill_name": group['Skill'].iloc[0],
        "subcategory": group['Subcategory'].iloc[0],
        "sub_area": group['Sub-area'].iloc[0],
        "tag": group['Tags'].iloc[0],
        "capabilities": group[['Capability_ID', 'Level', 'Capability_Description', 'Evidence_Type', 'Evidence_Metric']].rename(
            columns={
                'Capability_ID':'id',
                'Level':'level',
                'Capability_Description':'description',
                'Evidence_Type':'evidence_type',
                'Evidence_Metric':'evidence_metric'
            }
        ).to_dict(orient='records'),
        "what_it_covers": group['What it covers'].iloc[0]
    }

with open('SFIA_Capabilities_Full.json', 'w') as f:
    json.dump(sfia_json, f, indent=2)