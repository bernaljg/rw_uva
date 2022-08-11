#!/bin/bash

#SBATCH -o myjob.ubert_mlm_preds.out
#SBATCH -e myjob.ubert_mlm_preds.err

source myconda
mamba activate rw_uva1
python get_synonym_predictions.py /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/aui_vec/umls-vocab.txt /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/1_UMLS_ONLY/train_sp/out_all_correct_metric_from_32/checkpoint-290020_2/ ubert_mlm_290020_2 /data/Bodenreider_UMLS_DL/Interns/Bernal/umls2020AB_sapbert_edges.p