import pandas as pd

sfia_mapping = pd.read_csv('../SFIA_Development_and_Implementation/SFIA_Development_and_Implementation.csv')
capabilities = pd.read_csv('Capability_Evidence_Development_and_Implementation.csv')
sfia_mapping.rename(columns={'Code': 'SFIA_Code'}, inplace=True)
if 'Levels' in sfia_mapping.columns:
    sfia_mapping = sfia_mapping.drop(columns=['Levels'])
sfia_mapping = sfia_mapping.loc[:, ~sfia_mapping.columns.str.contains('^Unnamed')]

full_df = pd.merge(
    sfia_mapping, 
    capabilities, 
    on='SFIA_Code', 
    how='inner'
)

full_df = full_df[['Tags','Subcategory','Sub-area','SFIA_Code','Skill','Capability_ID',
                   'Capability_Description','Evidence_Type','Evidence_Metric','What it covers']]

full_df.to_csv('SFIA_Capabilities_Full.csv', index=False)