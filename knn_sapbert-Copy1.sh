#!/bin/bash

#SBATCH -o myjob.knn_sapbert.out
#SBATCH -e myjob.knn_sapbert.err

source myconda
mamba activate rw_uva1
python get_knn_sorted.py sapbert