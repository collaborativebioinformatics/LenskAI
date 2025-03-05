# LenskAI

"Lenski-esque AI competition trials with validated assertion databases"

# Contributors

- Rorry Brenner
- Peng Qiu
- Nanami Kubota
- Anshika Gupta
- Alicja Gluszko
- Jędrzej Kubica

# Aim

The aim of the project was to examine a Lenski-Esque experiment of evolving graph neural networks.  

## Introduction

Genomic medicine seeks to uncover molecular mechanisms responsible for human diseases. Large biological networks provide crucial information on complex relationships and interactions between biomolecules (e.g, genes or proteins) that underlie human diseases (https://doi.org/10.1038/nrg2918 ; https://doi.org/10.1093/bioadv/vbae099). Historically, and even today, experimental identification of genes involved in disease is expensive and time-consuming, often requiring extensive mouse and clinical studies. Conversely, network-based computational methods provide a way to model leverage biological networks to analyze genetype-phenotype associations. GeneDRAGGN [5] is a graph neural network for disease gene prioritization that leverages protein-protein interactions and disease-gene associations. In this study, we aimed at evolving the geneDRAGGN architecture in a Lenski-esque manner. Specifically, we started with the original architectures as the neural network “genotypes” and then “evolved” them by flipping individual components of one network to be components of the other networks. We selected those combinations that had the highest accuracy (on the test dataset provided by the authors) and let those architectures “survive” to the next iteration of training to then only continue “evolving”. We took the initial step of training a suite of networks of two types: 1) the architectures provided by the authors with replaced convolutional blocks; 2) an architecture where each layer was composed of a set of blocks of several types which were then averaged as the decision from that block.

# Methods and implementation

## Workflow

<img width="1066" alt="Screenshot 2025-03-05 at 1 12 12 PM" src="https://github.com/user-attachments/assets/dfb71ff2-cdba-49ec-8c96-fe2d501c1434" />


**Step 1: Preprocessing data** 

We used the data and preprocessing pipeline as outlined in the geneGRAGNN Github repository (https://github.com/geneDRAGNN/geneDRAGNN/blob/main/data/Readme.md)

Preprocessing pipeline: (https://github.com/geneDRAGNN/geneDRAGNN/blob/main/data_preprocessing/README.md)

In short, the following scripts were used for preprocessing the data:
- **import_dgn.py** - Imports the Disease Gene Network data and processes it to provide gene disease association scores and evidence index scores.
- **import_gdc.py** - Imports the National Institute of Health: Genomic Data Commons data for a specific disease and processes it to provide mutation features.
- **import_hpa.py** - Imports the Human Protein Atlas data and processes it to provide features based on genetic and RNA expression data. 
- **import_string.py** - Imports the STRING data for protein-protein interaction and processes it to provide the edge list and edge list features. 
- **create_node2vec_embeddings.py** - Applies an optimized node2vec to the target edgelist to create embeddings


**Step 2: Generate final input data**

main_data_pipeline.ipynb
Conducts the full data processing from start to finish by importing the features, edges and labels separately and providing the necessary operations to make the final datasets.


We ran the following experiments for evolving our GNN models:

![experiment1](Experiment1.png)

![experiment2](Experiment2.png)

![experiment3](Experiment3.png)

This workflow can be applied to other biological datasets available on the National Cancer Institute's Genomic Data Commons. 

During the hackathon we focused on the original geneDRAGNN dataset on lung cancer data.  We tried to using an additional brain cancer dataset from DisGeNET (https://doi.org/10.1093/nar/gkw943), however, the computational requirements exceeded our resources (even on a cloud infrastructure). 

### Other useful data sources

Protein-gene databases: Human Protein Atlas data, STRING, BioGRID, IntAct, Reactome

Genomic-disease data: Genomic Data Commons data for brain cancer data to provide mutation features. 

Disease–gene associations: DisGeNET

Useful script for downloading the DisGeNET data: https://github.com/dhimmel/disgenet

run `disgenet/disgenet.ipynb` to get all_gene_disease_associations.txt => All Gene Disease associations in DisGeNET


**PyTorch Geometric** GNN architectures to select from: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html

# Results

![image](https://github.com/user-attachments/assets/9affc6e7-e356-4d2b-b81b-481b5163a952)

<img width="894" alt="Screenshot 2025-03-05 at 3 24 38 PM" src="https://github.com/user-attachments/assets/f0d79dc6-d167-4e2e-8fb6-ea984b12f87f" />




# Discussion

Large biological networks such as protein-protein interaction networks or disease-gene association networks provide essential information about relationships and interactions between biomolecules. With the growing amount of such data, new bioinformatics approaches are needed. We conducted a project at the CMU / DNAnexus Hackathon 2025 to examine how we can leverage graph neural networks to extract biological insights from network-type of data. During the hackathon we aimed at training and "evolving" a graph neural network geneDRAGGN [5] for disease gene prioritization using public protein-protein interaction data (STRING database) and disease-gene association data (DisGeNET database). Our results suggest that imputing such large biological networks into graph neural networks is challenging due to computational requirements of such algorithms. We leveraged a cloud infrastructe, however, due to the size of the public data, we were not able to actually train the network.

Our preliminary results present a unique set of top10 genes that predict worse disease outcome in lung cancer patients.  

Looking forward, we plan to explore possible solutions to the challenge of imputing large biological networks into graph neural networks. In addition, we emphasize the importance of developing algorithms for reducing the size of network files so that the computation is more memory-efficient, yet the biological information is not removed.

## Special thanks to the Organizers of the CMU / DNAnexus Hackathon 2025!

### Installation and setup
    ## Setup and install
    Everything should be in the geneGRAGNN folder.

        pip install -r requirements.txt

    ## Run

        cd models
        CUDA_VISIBLE_DEVICES=0 python train_gnn_model_hackathon.py 

### DNANexus

    - Login to project
    - Click Project name
    - At top of screen click Tools->Jupyter lab
    - + new Jupyter Lab
        - Select project
        - Instance Type mem3_ssd1_gpu_x64
        - Duration 2 hours
    - Wait for initializing to complete
    - Go into the with "Open"
    - Other -> terminal

Then once you are there from terminal issue the following commands:

    git clone https://github.com/collaborativebioinformatics/LenskAI.git
    cd LenskAI/geneDRAGNN
    dx download data.tar.gz
    tar -xzvf data.tar.gz 
    cd models
    ENVNAME=ENVgene && python -m venv $ENVNAME && source $ENVNAME/bin/activate
    pip install --upgrade pip
    pip install numpy pandas tqdm torch pytorch_lightning torch_geometric wandb scikit-learn
    python train_gnn_model_hackathon.py "testName"
    

# References

1. Lenski RE. Twice as natural. Nature [Internet]. 2001 Nov 15;414(6861):255. Available from: http://dx.doi.org/10.1038/35104715
2. Mastropietro A, De Carlo G, Anagnostopoulos A. XGDAG: explainable gene-disease associations via graph neural networks. Bioinformatics [Internet]. 2023 Aug 1;39(8). Available from: http://dx.doi.org/10.1093/bioinformatics/btad482
3. Yan R, Islam MT, Xing L. Deep representation learning of protein-protein interaction networks for enhanced pattern discovery. Sci Adv [Internet]. 2024 Dec 20;10(51). Available from: http://dx.doi.org/10.1126/sciadv.adq4324
4. Stear BJ, Mohseni Ahooyi T, Simmons JA, Kollar C, Hartman L, Beigel K, et al. Petagraph: A large-scale unifying knowledge graph framework for integrating biomolecular and biomedical data. Sci Data [Internet]. 2024 Dec 18;11(1):1338. Available from: http://dx.doi.org/10.1038/s41597-024-04070-w
5. A. Altabaa, D. Huang, C. Byles-Ho, H. Khatib, F. Sosa and T. Hu, "geneDRAGNN: Gene Disease Prioritization using Graph Neural Networks," 2022 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), Ottawa, ON, Canada, 2022, pp. 1-10, doi: 10.1109/CIBCB55180.2022.9863043.
6. Chick, J. M., Munger, S. C., Simecek, P., Huttlin, E. L., Choi, K., Gatti, D. M., Raghupathy, N., Svenson, K. L., Churchill, G. A., & Gygi, S. P. (2016). Defining the consequences of genetic variation on a proteome-wide scale. Nature, 534(7608), 500–505.
7. Eizenga, J. M., Novak, A. M., Sibbesen, J. A., Heumos, S., Garrison, E., Sir’en, J., & Paten, B. (2020). Pangenome graphs. Annual Review of Genomics and Human Genetics, 21, 139–162.
8. Garrison, E., Sir’en, J., Novak, A. M., Hickey, G., Eizenga, J. M., Dawson, E. T., Jones, W., Garg, S., Markello, C., Lin, M. F., Paten, B., & Durbin, R. (2018). Variation graph toolkit improves read mapping by representing genetic variation in the reference. Nature Biotechnology, 36(9), 875–879.
9. Heath, A. P., Ferretti, V., Agrawal, S., An, M., Angelakos, J. C., Arya, R., Bajari, R., Baqar, B., Barnowski, J. H., Burt, J., & others. (2021). The NCI genomic data commons. Nature Genetics, 53(3), 257–262.
10. Piñero, J., Ramı́rez-Anguita, J. M., Saüch-Pitarch, J., Ronzano, F., Centeno, E., Sanz, F., & Furlong, L. I. (2020). The DisGeNET knowledge platform for disease genomics: 2019 update. Nucleic Acids Research, 48(D1), D845–D855.
11. Pontén, F., Jirström, K., & Uhlen, M. (2008). The Human Protein Atlas—a tool for pathology. The Journal of Pathology, 216(4), 387–393. https://doi.org/https://doi.org/10.1002/path.2440
12. Suhre, K., Shin, S.-Y., Petersen, A.-K., Mohney, R. P., Meredith, D., W"agele, B., Altmaier, E., CARDIoGRAM, Deloukas, P., Erdmann, J., Grundberg, E., Hammond, C. J., de Angelis, M. Hr., Kastenm"uller, G., K"ottgen, A., Kronenberg, F., Mangino, M., Meisinger, C., Meitinger, T., … Gieger, C. (2011). Human metabolic individuality in biomedical and pharmaceutical research. Nature, 477(7362), 54–60.
13. Sun, B. B., Maranville, J. C., Peters, J. E., Stacey, D., Staley, J. R., Blackshaw, J., Burgess, S., Jiang, T., Paige, E., Surendran, P., Oliver-Williams, C., Kamat, M. A., Prins, B. P., Wilcox, S. K., Zimmerman, E. S., Chi, A., Bansal, N., Spain, S. L., Wood, A. M., … Butterworth, A. S. (2018). Genomic atlas of the human plasma proteome. Nature, 558(7708), 73–79.
14. Szklarczyk, D., Gable, A. L., Nastou, K. C., Lyon, D., Kirsch, R., Pyysalo, S., Doncheva, N. T., Legeay, M., Fang, T., Bork, P., & others. (2021). The STRING database in 2021: customizable protein–protein networks, and functional characterization of user-uploaded gene/measurement sets. Nucleic Acids Research, 49(D1), D605–D612.
