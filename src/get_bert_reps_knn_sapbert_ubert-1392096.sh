#!/bin/bash

#SBATCH -o myjob.rep_sapbert_ubert_1392096.out
#SBATCH -e myjob.rep_sapbert_ubert_1392096.err

source myconda
mamba activate rw_uva1
python get_bert_representations_sorted_tokenizer.py cambridgeltl/SapBERT-from-PubMedBERT-fulltext /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/6_SAPBERT_SP/train/out_from_sapbert_from_3/checkpoint-1392096/ sapbert_ubert_1392096
python get_knn_sorted.py sapbert_ubert_1392096