"""
Script to evaluate kNN retrieval using either pre-computed vectors or a BERT model available in
the Huggingface model library or saved locally.

Inputs

- Original AUI Set: Set of AUIs to be used as the original ontology (the ontology to which new
AUIs will be attached).

- New AUI Set: Set of AUIs to be attached to "original" ontology.

- K: Number of nearest neighbors to extract from original ontology for
each new term.

- UMLS Directory: Path to the 'META' directory which stores UMLS tables.

- UMLS Version: Version of UMLS which contains all the AUIs referred to by the AUI sets.

- Output Directory: Path to the directory used for saving output files.

- Vector Dictionary (Optional): Python dictionary mapping AUIs or Strings to numpy vectors.

- BERT Model (Optional): If no vector dictionary, BERT model is necessary to extract representations

- TODO: New Term Set (Optional): If terms are not AUIs they can still be linked to the original ontology using only strings.

Outputs

- Set of K nearest neighbors from original ontology for each new AUI or term.
- AUI Based Recall @ 1,5,10,50,100,200,1000,2000
- CUI Based Recall @ 1,5,10,50,100,200,1000,2000
"""

import os
import sys
from UMLS import UMLS


def main():
    umls_dir = sys.argv[1]
    umls_version = sys.argv[2]
    original_auis_filename = sys.argv[3]
    new_auis_filename = sys.argv[4]
    k = int(sys.argv[5])
    output_dir = sys.argv[6]
    num_retriever_names = int(sys.argv[7])
    retriever_names = sys.argv[8:8 + num_retriever_names]
    classifier_name = sys.argv[9 + num_retriever_names]
    candidates_to_classify = int(sys.argv[10 + num_retriever_names])
    add_gold_candidates = eval(sys.argv[10 + num_retriever_names])
    dev_perc = eval(sys.argv[11 + num_retriever_names])
    test_perc = eval(sys.argv[12 + num_retriever_names])

    umls = UMLS(umls_dir, umls_version)
    umls.augment_umls(original_auis_filename,
                      new_auis_filename,
                      output_dir,
                      retriever_names,
                      k,
                      classifier_name,
                      candidates_to_classify,
                      add_gold_candidates,
                      dev_perc,
                      test_perc
                      )

if __name__ == "__main__":
    main()