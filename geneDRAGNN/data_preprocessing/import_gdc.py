import pandas as pd

def ImportGDC():
    gdc = pd.read_csv("../data/raw_data/gdc_luad_genes.csv")

    featureNameColumns = ['# SSM Affected Cases in Cohort', '# CNV Gain', '# CNV Loss']
    for i in featureNameColumns:
        # Extract numerator and denominator
        gdc[[i + '_num', i + '_den']] = gdc[i].str.replace(',', '', regex=True).str.extract(r'(\d+) / (\d+)')
        # Convert to float and compute ratio
        gdc[i] = gdc[i + '_num'].astype(float) / gdc[i + '_den'].astype(float)
        # Drop temporary columns
        gdc.drop(columns=[i + '_num', i + '_den'], inplace=True)

    # Drop unnecessary columns
    gdc.drop(['Symbol', 'Name', 'Cytoband', 'Type', 'Annotations', 'Survival'], axis=1, inplace=True)

    # Process '# SSM Affected Cases Across the GDC'
    gdc[['ssm_num', 'ssm_den']] = gdc['# SSM Affected Cases Across the GDC'].str.replace(',', '', regex=True).str.extract(r'(\d+) / (\d+)')
    gdc['# SSM Affected Cases Across the GDC'] = gdc['ssm_num'].astype(float) / gdc['ssm_den'].astype(float)
    gdc.drop(columns=['ssm_num', 'ssm_den'], inplace=True)

    # Rename columns
    gdc = gdc.rename({'# SSM Affected Cases in Cohort': 'nih_ssm_in_cohort', 
                      '# SSM Affected Cases Across the GDC': 'nih_ssm_across_gdc',
                      '# CNV Gain': 'nih_cnv_gain', 
                      '# CNV Loss': 'nih_cnv_loss', 
                      'Gene ID': 'ensembl', 
                      '# Mutations': 'nih_tot_mutations'}, axis=1)

    return gdc
