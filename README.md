# LenskAI

"Lenski-esque AI competition trials with validated assertion databases"

## Setup and install
Everything should be done in the XGDAG folder.

    cd XGDAG
    ENVNAME=ENVxgdag && python -m venv $ENVNAME && source $ENVNAME/bin/activate
    pip install scikit-learn torch_geometric rdkit seaborn torch

## Data

The data generated and provided in this repository are based on PPI data from [BioGRID](https://thebiogrid.org/) and Gene-Disease Associations from [DisGeNET](https://www.disgenet.org/). The original data can be dowloaded from the related websites. Part of the analysis relies on the set of all disease associations from DisGeNET. Given the size of this file, it needs to be manually downloaded from [here](https://drive.google.com/file/d/12cyI6ds0mKQI9mcRgaf0_9v8KDZHWpQR/view?usp=sharing) and placed in the ```Datasets``` folder.

Using the aformentioned data we built graphs available for use in the ```Graphs``` folder. The script ```CreateGraph.py``` was used for this purpose.

## Run

    CUDA_VISIBLE_DEVICES=0 python TrainerScript.py 

# Aim

The aim of the project is to examine how graph neural networks learn and provide valuable insights from a network-type of data. We plan to feed a graph neural network (GNN) with a large biological network (e.g., DisGeNET, PrimeKG, SPOKE, petagraph). Then change the network architecture and compete both networks agains each other to examine how GNN evolve.

Interesting use cases:
- background effects (secondary genes)
- subtyping of AD -> how many subtypes are there? 
- disease phenotype networks

# Contributors

- 

# Introduction

<img width="795" alt="Screenshot 2025-03-03 at 20 47 30" src="https://github.com/user-attachments/assets/19681a5d-355d-4b5f-8d18-508330dc073a" />

<img width="367" alt="Screenshot 2025-03-03 at 20 47 37" src="https://github.com/user-attachments/assets/39e8d1d0-e607-4281-a560-f0ee473e3ab6" />

# Methods

Extracting disease–gene associations from DisGeNET: https://github.com/dhimmel/disgenet

Protein-protein interaction databases: BioGRID, IntAct, Reactome, KEGG

**PyTorch Geometric** GNN architectures to select from: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

## Workflow

![workflow](https://github.com/user-attachments/assets/929e7817-d43d-4cf4-b5d3-271b6a573ed3)

# Results

# Discussion

# References

1. Lenski RE. Twice as natural. Nature [Internet]. 2001 Nov 15;414(6861):255. Available from: http://dx.doi.org/10.1038/35104715
2. Mastropietro A, De Carlo G, Anagnostopoulos A. XGDAG: explainable gene-disease associations via graph neural networks. Bioinformatics [Internet]. 2023 Aug 1;39(8). Available from: http://dx.doi.org/10.1093/bioinformatics/btad482
3. Yan R, Islam MT, Xing L. Deep representation learning of protein-protein interaction networks for enhanced pattern discovery. Sci Adv [Internet]. 2024 Dec 20;10(51). Available from: http://dx.doi.org/10.1126/sciadv.adq4324
4. Stear BJ, Mohseni Ahooyi T, Simmons JA, Kollar C, Hartman L, Beigel K, et al. Petagraph: A large-scale unifying knowledge graph framework for integrating biomolecular and biomedical data. Sci Data [Internet]. 2024 Dec 18;11(1):1338. Available from: http://dx.doi.org/10.1038/s41597-024-04070-w
