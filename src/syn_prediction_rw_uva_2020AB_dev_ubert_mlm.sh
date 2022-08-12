#!/bin/bash

#SBATCH -o myjob.ubert_mlm_preds_uva.out
#SBATCH -e myjob.ubert_mlm_preds_uva.err

source myconda
mamba activate rw_uva1
python get_synonym_predictions.py /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/aui_vec/umls-vocab.txt /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/1_UMLS_ONLY/train_sp/out_all_correct_metric_from_32/checkpoint-290020_2/ ubert_mlm_rw_uva_2020AB_dev_set_290020_2 /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/ubert_2020AB_dev_set.p