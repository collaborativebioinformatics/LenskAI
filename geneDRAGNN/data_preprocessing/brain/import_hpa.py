import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

def ImportHPA():
    hpa = pd.read_csv('./data_subset/hpa_gene_features.tsv', sep='\t').drop_duplicates(subset='Gene')

    identifiers = [
        "Gene",
        "Ensembl"
    ]
    discrete_features = [
        "Protein class",
        "Biological process",
        "Molecular function",
        "Disease involvement",
        "Subcellular location",
    ]
    continuous_features = [
        "Tissue RNA - cerebral cortex [NX]",
        "Tissue RNA - cerebellum [NX]",
        "Tissue RNA - olfactory region [NX]",
        "Tissue RNA - basal ganglia [NX]",
        "Tissue RNA - thalamus [NX]",
        "Tissue RNA - hypothalamus [NX]",
        "Tissue RNA - midbrain [NX]",
        "Tissue RNA - amygdala [NX]",
        "Tissue RNA - pons and medulla [NX]",
        "Tissue RNA - hippocampal formation [NX]",
        "Tissue RNA - spinal cord [NX]"
        # "Single Cell Type RNA - Mucus-secreting cells [NX]"
    ]

    hpa_features = hpa.iloc[:, hpa.columns.isin(identifiers+discrete_features+continuous_features)]

    for col in continuous_features:
        hpa_features[col] = (hpa_features[col] - hpa_features[col].mean()) / hpa_features[col].std()

    def explode(feature) :
        return feature.apply(lambda x: x.replace(' ', '').split(','))

    hpa_clean = hpa.fillna('')
    for ft in discrete_features :
        hpa_clean[ft] = explode(hpa_clean[ft])

    protein_class = hpa_clean["Protein class"].explode().unique()
    biological_process = hpa_clean["Biological process"].explode().unique()
    molecular_function = hpa_clean["Molecular function"].explode().unique()
    disease_involvement = hpa_clean["Disease involvement"].explode().unique()
    subcellular_location = hpa_clean["Subcellular location"].explode().unique()
    GO_features = np.concatenate([protein_class, biological_process, molecular_function, disease_involvement, subcellular_location])

    RowFeatures = pd.DataFrame(data = 0,index = hpa_clean['Ensembl'],columns=GO_features)
    counter = 0

    for index, row in RowFeatures.iterrows() :
        features = hpa_clean.iloc[counter][['Protein class', 'Biological process', 'Molecular function', 'Disease involvement', 'Subcellular location']].to_list()
        flattened = [item for sublist in features for item in sublist if item]
        for t in flattened :
            row[t] = 1
        counter +=1 

    n_comp = 100
    svd = TruncatedSVD(n_components = n_comp)
    svdModel = svd.fit(RowFeatures)
    visits_emb = svdModel.transform(RowFeatures)
    hpa = pd.DataFrame(data=visits_emb, index=RowFeatures.index).reset_index()
    hpa.columns = ['hpa_' + str(col) for col in hpa.columns]
    
    # Rename columns: remove [NX], convert to lowercase, replace spaces and hyphens with underscores
    hpa.columns = (
        hpa.columns.str.replace(r'\[NX\]', '', regex=True)  # Remove [NX]
        .str.lower()  # Convert to lowercase
        .str.replace(r'[\s\-]+', '_', regex=True)  # Replace spaces and hyphens with underscores
    )

    # Ensure 'Ensembl' column is renamed to 'ensembl'
    hpa = hpa.rename({'hpa_ensembl': 'ensembl'}, axis=1)

    return hpa