#!/bin/bash

#SBATCH -o myjob.out
#SBATCH -e myjob.err

source myconda
mamba activate rw_uva1
python extract_abstracts_pubmed.py