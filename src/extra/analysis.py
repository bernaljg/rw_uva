import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import pickle
import os
from tqdm import tqdm

def compute_scores(TP, TN, FP, FN):
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision, recall, f1 = 0, 0, 0
    
    if TP + FP > 0:
        precision = TP/(TP+FP)
        
    if TP + FN > 0:
        recall = TP/(TP+FN)
        
    if recall + precision > 0:
        f1 = 2*(recall * precision) / (recall + precision)
    
    return round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)


def load_pickle(pickle_fp):
    with open(pickle_fp, 'rb') as f:
        obj = pickle.load(f)
    return obj

#get model metrics of whole dataset
def get_model_metrics(scores):
    TP = scores["TP"].sum()
    TN = scores["TN"].sum()
    FP = scores["FP"].sum()
    FN = scores["FN"].sum()
    print('total pairs predicted: {}'.format(TP + TN + FP + FN))
    print('total positive pairs = {}'.format(TP + FN))
    print('total negative pairs = {}'.format(TN + FP))
    
    print('TP = {}, TN = {}, FN = {}, FP = {}'.format(TP, TN, FN, FP))
    return compute_scores(TP, TN, FP, FN)

def get_aui_recall_count(scores):
    """Creates new dataframe with columns being AUI_ID, AUI, recall, pos pairs count. 
    This is used to analyze worst performing recall per AUI while taking into consideration 
    the total number of positive pairs  

    Args:
        scores (dataframe): dataframe of AUI_ID, TP, FP, FN, TN
    """
    if os.path.isfile('recall_scores.csv'):
        return pd.read_csv('recall_scores.csv')
    unique_id2aui = load_pickle('UNIQUE_ID2AUI.PICKLE')
    ab_mrc_atoms = load_pickle('AB_MRC_ATOMS.PICKLE')
    recall_scores = pd.DataFrame(columns=['AUI_ID', 'AUI', 'CUI', 'SG', 'recall', 'pos_pairs'])
    
    with tqdm(total=len(unique_id2aui)) as pbar:
        for idx, row in scores.iterrows():
            # if idx > 10:
            #     break
        
            aui = unique_id2aui[row['AUI_ID']]
            # print(row['AUI_ID'])
            recall = 0
            total_pos = row['TP'] + row['FN']
            if total_pos != 0:  
                recall =  row['TP'] / (row['TP'] + row['FN'])
            recall_row = {"AUI_ID": row['AUI_ID'], "AUI": aui, 'CUI': ab_mrc_atoms[aui]['CUI'], 'SG': ab_mrc_atoms[aui]['SG'], "recall": recall, 'pos_pairs': row['TP'] + row['FN']}
            recall_scores = recall_scores.append(recall_row, ignore_index=True)
            pbar.update(1)
        
    recall_scores.to_csv('recall_score.csv')
    return recall_scores
    

#get recall of each aui, bin by increments of 10%
def analyze_recall(recall_scores):
    """Groups recall_scores into 10 bins. Then gets the number of auis in each bin, this tells us the distribution of recall 
    to determine if there exists singificant amount of poorly performing auis

    Args:
        recall_scores (dataframe):
    """
    recall_scores['recall'] = pd.cut(recall_scores['recall'], 10)
    recall_grouped = recall_scores.groupby(by=['recall'])
    print(recall_grouped.agg({'AUI_ID': ['count'], 'pos_pairs': ['mean']}))
    return recall_grouped

def analyze_sg(name, group):
    TP = group['TP'].sum()
    TN = group['TN'].sum()
    FP = group['FP'].sum()
    FN = group['FN'].sum()
    
    accuracy, precision, recall, f1 = compute_scores(TP, TN, FP, FN)
    return {"SG": name, "TP": TP, "TN": TN, "FP": FP, "FN": FN, "Accuracy": accuracy,
            "Recall": recall, "Precision": precision, "F1": f1}
    
    

    

#get total positive paris per aui

#create double double sided bar graph with y-axis-1 being recall and y-axis-2 being total positive
def main():
    scores_sg = pd.read_csv('scores_sg.csv')
    # sg_model_metrics = pd.DataFrame(columns=['SG', "TP", "TN", "FP", "FN", "Accuracy", "Recall", "Precision", "F1"])
    # sg_groups_df = scores_sg.groupby(['SG'])
    # for name, group in sg_groups_df:
    #     row = analyze_sg(name, group)
    #     sg_model_metrics = sg_model_metrics.append(row,ignore_index=True)
    # sg_model_metrics.to_csv("SG_Model_Metrics.csv")
    
    
    TP = scores_sg['TP'].sum()
    FP = scores_sg['FP'].sum()
    TN = scores_sg['TN'].sum()
    FN = scores_sg['FN'].sum()
    accuracy, precision, recall, f1 = compute_scores(TP, TN, FP, FN)
    
    print('accuracy = {},precision = {}, recall = {}, f1 = {}'.format(accuracy, precision, recall, f1))
    # recall_scores = get_aui_recall_count(scores)
    # recall_grouped = analyze_recall(recall_scores)
    # analysis_sg(recall_grouped)
    
    
    

if __name__ == "__main__":
    main()