
gdc <- read.delim("/Users/kubotan/Documents/github/LenskAI/geneDRAGNN/data_subset/frequently-mutated-genes.2025-03-04_braincancer.tsv")

gdc_mod <- gdc %>%
  mutate(
    SSM_Affected_Cases_in_Cohort = paste0(num_cohort_ssm_affected_cases," / ", num_cohort_ssm_cases, " (",cohort_ssm_affected_cases_percentage,"%)"),
    CNV_Gain = paste0(num_cohort_cnv_gain_cases, " / ", num_cohort_cnv_cases," (",	cohort_cnv_gain_cases_percentage,"%)"),
    CNV_Loss = paste0(num_cohort_cnv_loss_cases, " / ", num_cohort_cnv_cases," (",	cohort_cnv_loss_cases_percentage,"%)"),
    SSM_Affected_Cases_Across_the_GDC = paste0(num_gdc_ssm_affected_cases," / ",	num_gdc_ssm_cases, " (",gdc_ssm_affected_cases_percentage,"%)")
    )

write.csv(gdc_mod, "/Users/kubotan/Documents/github/LenskAI/geneDRAGNN/data_subset/frequently-mutated-genes.2025-03-04_braincancer.csv", row.names = FALSE)


gdc_luac <- read.csv("/Users/kubotan/Documents/github/lenskai_NK/data/raw_data/gdc_luad_genes.csv")
