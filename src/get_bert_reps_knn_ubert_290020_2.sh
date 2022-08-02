#!/bin/bash

#SBATCH -o myjob.rep_ubert_290020_2.out
#SBATCH -e myjob.rep_ubert_290020_2.err

source myconda
mamba activate rw_uva1
python get_bert_representations_sorted_tokenizer.py /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/aui_vec/umls-vocab.txt /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/1_UMLS_ONLY/train_sp/out_all_correct_metric_from_32/checkpoint-290020_2/ ubert_mlm_290020_2
python get_knn_sorted.py ubert_mlm_290020_2