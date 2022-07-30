#!/bin/bash

#SBATCH -o myjob.sapbert_ubert_mlm_preds_rw_uva_subset.out
#SBATCH -e myjob.sapbert_ubert_mlm_preds_rw_uva_subset.err

source myconda
mamba activate rw_uva1
python get_synonym_predictions.py cambridgeltl/SapBERT-from-PubMedBERT-fulltext /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/6_SAPBERT_SP/train/out_from_sapbert_from_3/checkpoint-1392096/ sapbert_ubert_rw_uva_subset_1392096 /data/Bodenreider_UMLS_DL/Interns/Bernal/umls2020AB_sapbert_edges_subset.p