#!/bin/bash

#SBATCH -o myjob.ubert_mlm.out
#SBATCH -e myjob.ubert_mlm.err

source myconda
mamba activate rw_uva1
python get_bert_representations_sorted_tokenizer.py /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/tokenizers/umls_pubmed_pmc-vocab.txt /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/1_UMLS_ONLY/train_mlm/output_all/checkpoint-3500/ ubert_mlm