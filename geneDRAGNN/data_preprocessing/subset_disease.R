
setwd("/Users/kubotan/Documents/github/LenskAI/geneDRAGNN")

all_disease <- read.delim("./data/all_gene_disease_associations.tsv")

library(tidyverse)
library(stringr)

bone_disease <- all_disease %>%
  dplyr::filter(str_detect(diseaseName, "bone"))

brain_disease <- all_disease %>%
  dplyr::filter(str_detect(diseaseName, "glio"))

write.csv(bone_disease, "./data/bone_disease_associations.csv")
write.csv(brain_disease, "./data/brain_disease_associations.csv")
