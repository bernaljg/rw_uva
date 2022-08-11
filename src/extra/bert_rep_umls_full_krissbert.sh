#!/bin/bash

#SBATCH -o myjob.out
#SBATCH -e myjob.err

source myconda
mamba activate rw_uva1
python get_bert_representations_sorted.py microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL krissbert