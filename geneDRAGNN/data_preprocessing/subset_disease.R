
setwd("/Users/kubotan/Documents/github/LenskAI/geneDRAGNN/data_preprocessing/brain/data_subset")

all_disease <- read.delim("./data_subset/all_gene_disease_associations.tsv")

library(tidyverse)
library(stringr)

bone_disease <- all_disease %>%
  dplyr::filter(str_detect(diseaseName, "bone"))%>%
  select(geneSymbol, EI, score) %>%
  rename(Gene = geneSymbol) %>%
  distinct()

brain_disease <- all_disease %>%
  dplyr::filter(str_detect(diseaseName, "glio")) %>%
  select(geneSymbol, EI, score) %>%
  rename(Gene = geneSymbol) %>%
  distinct()

write.csv(bone_disease, "./bone_disease_associations.csv", row.names = FALSE)
write.csv(brain_disease, "./brain_disease_associations.csv", row.names = FALSE)
