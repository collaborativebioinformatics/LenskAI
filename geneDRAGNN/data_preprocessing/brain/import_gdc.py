import pandas as pd

def ImportGDC():
    gdc = pd.read_csv("./data_subset/frequently-mutated-genes.2025-03-04_braincancer.csv")

    featureNameColumns = ['SSM_Affected_Cases_in_Cohort', 'CNV_Gain', 'CNV_Loss']
    for i in featureNameColumns:
        # Extract numerator and denominator
        gdc[[i + '_num', i + '_den']] = gdc[i].str.replace(',', '', regex=True).str.extract(r'(\d+) / (\d+)')
        # Convert to float and compute ratio
        gdc[i] = gdc[i + '_num'].astype(float) / gdc[i + '_den'].astype(float)
        # Drop temporary columns
        gdc.drop(columns=[i + '_num', i + '_den'], inplace=True)

    # Drop unnecessary columns
    gdc.drop(['symbol', 'name', 'cytoband', 'type', 'annotations'], axis=1, inplace=True)

    # Process '# SSM Affected Cases Across the GDC'
    gdc[['ssm_num', 'ssm_den']] = gdc['SSM_Affected_Cases_Across_the_GDC'].str.replace(',', '', regex=True).str.extract(r'(\d+) / (\d+)')
    gdc['SSM_Affected_Cases_Across_the_GDC'] = gdc['ssm_num'].astype(float) / gdc['ssm_den'].astype(float)
    gdc.drop(columns=['ssm_num', 'ssm_den'], inplace=True)

    # Rename columns
    gdc = gdc.rename({'SSM_Affected_Cases_in_Cohort': 'nih_ssm_in_cohort', 
                      'SSM_Affected_Cases_Across_the_GDC': 'nih_ssm_across_gdc',
                      'CNV_Gain': 'nih_cnv_gain', 
                      'CNV_Loss': 'nih_cnv_loss', 
                      'gene_id': 'ensembl', 
                      'num_mutations': 'nih_tot_mutations'}, axis=1)

    return gdc
