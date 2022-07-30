#!/usr/bin/env python
# coding: utf-8

# In[1]:


import _pickle as pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import copy
import glob
import gc


# In[2]:


pd.set_option('precision',2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('max_rows',200)


# In[3]:


gc.disable()


# In[4]:


full_df = []

exp_list = glob.glob('/data/Bodenreider_UMLS_DL/Interns/Bernal/UMLS2020AB*2000*')

print(exp_list)

for filename in tqdm(exp_list):
    file = open(filename,'rb')
    full_df.append(pickle.load(file))
    file.close()


# In[5]:


umls2020AB_df = full_df[0]


# In[6]:


for df in full_df[1:]:
    nn_columns = df.filter(regex='.*NN.*').columns
    for col in nn_columns:
        umls2020AB_df[col] = df[col]


# In[7]:


del full_df


# In[8]:


umls2020AB_df.columns


# In[9]:


for col in umls2020AB_df.filter(regex='lexlm.*').columns:
    umls2020AB_df.drop(col,axis=1)


# In[10]:


nearest_neighbors_auis = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Bernal/lex_lm_2000-NN.p','rb'))
nearest_neighbors_dist = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Bernal/lex_lm_2000-NN_dist.p','rb'))
nearest_neighbors_auis = [auis for auis in nearest_neighbors_auis]

original_umls_2020, new_umls_2020 = pickle.load(open('aui_string_map_UMLS2020_update.p','rb'))

new_umls_2020 = [x[0] for x in new_umls_2020]
new_umls_2020 = pd.DataFrame(new_umls_2020,columns=['auis'])
new_umls_2020['lexlm_2000-NN_auis']  = nearest_neighbors_auis
new_umls_2020['lexlm_2000-NN_dist']  = list(nearest_neighbors_dist)

umls2020AB_df = umls2020AB_df.merge(new_umls_2020,on='auis',how='inner')

query_synonym_auis = list(umls2020AB_df['2020AA_synonyms'])
nearest_neighbors_auis = umls2020AB_df['lexlm_2000-NN_auis']

#Calculating Recall @ 1,5,10,50,100
recall_array = []
# closest_dist_true = []
# closest_dist_false = []

for true_syn, top100 in tqdm(zip(query_synonym_auis, nearest_neighbors_auis)):
    
    true_syn = set(true_syn)
    
    if len(true_syn) > 0:
        recalls = []

        for n in [1,5,10,50,100,200,500,1000,2000]:

            topn = set(top100[:n])
            true_pos = topn.intersection(true_syn)

            recalls.append(len(true_pos)/len(true_syn))

        recall_array.append(recalls)
#         closest_dist_true.append([top100_dist[0], np.mean(top100_dist)])
    else:
        recalls = []

        recall_array.append(recalls)
#         closest_dist_false.append([top100_dist[0], np.mean(top100_dist)])

umls2020AB_df['lexlm_2000-NN_recall'] = recall_array


# In[11]:


for recall_col in umls2020AB_df.filter(regex='.*recall.*').columns:
    print(recall_col)
    recall_array = list(umls2020AB_df[recall_col].values)


aui_info = []

with open('/data/Bodenreider_UMLS_DL/UMLS_VERSIONS/2020AB-ACTIVE/META/MRCONSO.RRF','r') as fp:
    
    for line in fp.readlines():
        line = line.split('|')
        cui = line[0]
        aui = line[7]
        string = line[-5]
        
        aui_info.append({'AUI':aui, 'CUI':cui, 'STR':string})
        
cui2sg = {}

with open('/data/Bodenreider_UMLS_DL/UMLS_VERSIONS/2020AB-ACTIVE/META/MRSTY.RRF','r') as fp:
    
    for line in fp.readlines():
        line = line.split('|')
        cui = line[0]
        sg = line[3]
        cui2sg[cui] = sg
        
original_umls = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/INTERSECT_AUI2ID.PICKLE','rb'))
new_auis = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/UNIQUE_AUI2ID.PICKLE','rb'))

aui_vecs  = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/AUI2LAYER.PICKLE','rb'))

all_2020_auis = set(original_umls.keys()).union(new_auis.keys())

cui2aui = {}
aui2cui = {}
aui2str = {}
aui2sg = {}

cui_sg = []
cui_aui = []

for tup in aui_info:
    aui = tup['AUI']
    
    if aui in all_2020_auis:        
        cui = tup['CUI']
        string = tup['STR']
        sg = cui2sg[cui]

        auis = cui2aui.get(cui, [])
        auis.append(aui)
        cui2aui[cui] = auis

        aui2cui[aui] = cui
        aui2str[aui] = string
        aui2sg[aui] = sg

        cui_sg.append((cui, sg))
        cui_aui.append((cui, aui))
        
semgroups = pd.read_csv('SemGroups.txt',sep='|',header=None)

semtype2sg = {}

for i, row in semgroups.iterrows():
    
    st = row[3]
    sg = row[1]
    
    semtype2sg[st] = sg
    
cuis = []
sts = []

for aui in umls2020AB_df.auis:
    
    cuis.append(aui2cui[aui])
    sts.append(aui2sg[aui])
    
umls2020AB_df['cuis'] = cuis
umls2020AB_df['sem_types'] = sts
umls2020AB_df['sem_groups'] = [semtype2sg[st] for st in sts]


# In[13]:


nearest_neighbors_auis = umls2020AB_df['lexlm_2000-NN_auis']

nearest_neighbors_strings = []

for nn_auis in tqdm(nearest_neighbors_auis):
    nn_strings = [aui2str[aui] for aui in nn_auis]
    
    nearest_neighbors_strings.append(nn_strings)
    
umls2020AB_df['lexlm_2000-NN_strings'] = nearest_neighbors_strings


# In[14]:


for recall_col in umls2020AB_df.filter(regex='.*recall.*').columns:
    atn_recall = []
    for i,row in tqdm(umls2020AB_df.iterrows()):
        recalls = row[recall_col]

        if len(recalls) > 0:
            atn_recall.append(recalls)
        else:
            atn_recall.append([None for i in [1,5,10,50,100,200,500,1000,2000]])
        
    recall_col_name = recall_col.split('_2000')[0]
    
    for index,n in tqdm(enumerate([1,5,10,50,100,200,500,1000,2000])): 
        umls2020AB_df['R@{}_{}'.format(n,recall_col_name)] = np.array(atn_recall)[:,index] 


# In[15]:


len(umls2020AB_df.filter(regex='R@.*').columns)


# In[16]:


umls2020AB_df.columns


# In[17]:


umls2020AB_df.head()


# In[18]:


pd.set_option('max_colwidth',500)


# In[19]:


string_cols = umls2020AB_df.filter(regex='.*NN_strings').columns


# In[20]:


string_cols


# In[21]:


for col in string_cols:
        
    errors = []
    
    for i, row in tqdm(umls2020AB_df.iterrows(),total=len(umls2020AB_df)):
        syns = set(row['synonym_strings'])
        if len(syns) > 0:      
            pred_syns = set(row[col])
            missed_syns = syns.difference(pred_syns)
            errors.append('|||'.join([str(s) for s in missed_syns]))
        else:
            errors.append([])
            
    umls2020AB_df[col.split('_2000')[0]+'_errors'] = errors


# In[66]:


f = open('/data/Bodenreider_UMLS_DL/Interns/Bernal/UMLS2020AB_Full_NN_DataFrame_UpToSAPBERT_UBERT.p','wb')


# In[ ]:


pickle.dump(umls2020AB_df, f)




