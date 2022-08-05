#!/bin/bash

#SBATCH -o augment_umls_test.myjob.out
#SBATCH -e augment_umls_test.myjob.err

#source myconda
#mamba activate rw_uva1
python augment_UMLS_generate_cls_dataset.py /data/Bodenreider_UMLS_DL/UMLS_VERSIONS 2020AB /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/2020AA_split.sem_group_stratified.original_auis.p /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/2020AA_split.sem_group_stratified.new_auis.p 2000 /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/output 2 scui_lui cambridgeltl/SapBERT-from-PubMedBERT-fulltext None 100 False 1000 2000
python augment_UMLS_generate_cls_dataset.py /data/Bodenreider_UMLS_DL/UMLS_VERSIONS 2020AB /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/2020AA_split.sem_group_stratified.original_auis.p /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/2020AA_split.sem_group_stratified.new_auis.p 2000 /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/output 2 scui_lui cambridgeltl/SapBERT-from-PubMedBERT-fulltext None 100 True 1000 2000