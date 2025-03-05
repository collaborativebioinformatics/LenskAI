import pandas as pd

def ImportDGN():
    dgn = pd.read_csv("./data_subset/brain_disease_associations.csv")
    dgn_dict = pd.read_csv("./data_subset/gda_dictionary.csv", index_col=None)

    score_threshold = 0.02
    ei_threshold = 0.7

    dgn = dgn[['Gene', 'EI', 'score']]
    dgn = dgn.loc[dgn['score'] >= score_threshold]
    dgn = dgn.loc[dgn['EI'] > ei_threshold]
    dgn.rename({'score':'gda_score'}, axis=1, inplace=True)
    dgn = dgn.merge(dgn_dict, on="Gene").drop(['Gene'], axis=1)
    dgn['gda_score'] = 1

    return dgn[['ensembl', 'gda_score']]