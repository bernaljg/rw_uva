#!/bin/bash

#SBATCH -o myjob.knn_pubmedbert.out
#SBATCH -e myjob.knn_pubmedbert.err

source myconda
mamba activate rw_uva1
python get_knn_sorted.py pubmedbert