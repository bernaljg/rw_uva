import sys
from pathlib import Path
import os
path = Path(os.getcwd())
sys.path.append(path.parent.absolute())
import os
import numpy as np
import csv
import pickle
import pandas as pd

def load_pickle(pickle_fp):
    with open(pickle_fp, 'rb') as f:
        obj = pickle.load(f)
    return obj

def main():
    unique_aui2id = load_pickle('UNIQUE_AUI2ID.PICKLE')
    intersect_aui2id = load_pickle('INTERSECT_AUI2ID.PICKLE')
    intersect_sg2id = load_pickle('INTERSECT_SG2ID.PICKLE')
    unique_sg2id = load_pickle('UNIQUE_SG2ID.PICKLE')
    aui_info = load_pickle("AUI_INFO.PICKLE")
    sg = int()
    idx_id = int()
    ab_mrc_atoms = load_pickle("AB_MRC_ATOMS.PICKLE")
    
    scores_sg = pd.read_csv('scores_sg.csv')
    # for idx, row in scores_sg.iterrows():
        
    #     info = aui_info[row['AUI_ID']]
    #     sg = info['SG']
    #     expected_pairs = len(intersect_sg2id[sg]) + len(unique_sg2id[sg][info['IDX']+1:])
    #     observed_pairs = row['TP'] + row['TN'] + row['FP'] + row['FN']
    #     if expected_pairs != observed_pairs:
    #         print('unequal pairs')
    #         break
        
        #total auis
    total = 0
    for sg, id_arr in intersect_sg2id.items():
        total += len(id_arr)
    for sg, id_arr in unique_sg2id.items():
        print(sg)
        
    print("number of sg in uniquesg2id = {}".format(len(unique_sg2id)))
    
    
    
if __name__ == '__main__':
      main()
