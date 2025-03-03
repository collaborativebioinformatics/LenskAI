# LenskAI

"Lenski-esque AI competition trials with validated assertion databases"

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
