#!/bin/bash

#SBATCH -o augment_umls_test.myjob.out
#SBATCH -e augment_umls_test.myjob.err

#source myconda
#mamba activate rw_uva1
python augment_UMLS.py /data/Bodenreider_UMLS_DL/UMLS_VERSIONS 2020AB /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/vishesh_dataset_original_auis.p /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/vishesh_dataset_new_auis.p 2000 /data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/output 4 scui scui_lui scui_cui cambridgeltl/SapBERT-from-PubMedBERT-fulltext