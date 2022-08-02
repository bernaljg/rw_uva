#!/bin/bash

#SBATCH -o myjob.knn_krissbert.out
#SBATCH -e myjob.knn_krissbert.err

source myconda
mamba activate rw_uva1
python get_knn_sorted.py ubert_mlm