#!/bin/bash

#SBATCH -o myjob.ubert_mlm.out
#SBATCH -e myjob.ubert_mlm.err

source myconda
mamba activate rw_uva1
python get_bert_representations_sorted_tokenizer.py cambridgeltl/SapBERT-from-PubMedBERT-fulltext /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/6_SAPBERT_SP/train/out_from_sapbert/checkpoint-58004/ sapbert_ubert