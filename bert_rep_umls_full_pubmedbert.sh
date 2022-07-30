#!/bin/bash

#SBATCH -o pubmedbert.myjob.out
#SBATCH -e pubmedbert.myjob.err

source myconda
mamba activate rw_uva1
python get_bert_representations_sorted.py microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext pubmedbert