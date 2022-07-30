#!/bin/bash

#SBATCH -o myjob.ubert_mlm_preds_uva.out
#SBATCH -e myjob.ubert_mlm_preds_uva.err

source myconda
mamba activate rw_uva1
python get_synonym_predictions.py /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/tokenizers/umls_pubmed_pmc-vocab.txt /data/Bodenreider_UMLS_DL/thilini/EXPERIMENTS/1_UMLS_ONLY/train_sp/out_all_correct_metric_from_32/checkpoint-551038_2/ ubert_mlm_2_uva_test /data/Bodenreider_UMLS_DL/Interns/Bernal/uva_test_subset.p