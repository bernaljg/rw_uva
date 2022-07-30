import _pickle as pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import copy
import glob
import gc
import ipdb
import time


# In[2]:


# In[ ]:


full_df = []

exp_list = glob.glob('/data/Bodenreider_UMLS_DL/Interns/Bernal/UMLS2020AB_*.2000-NN_DataFrame.p')

print(exp_list)

for filename in tqdm(exp_list):
    file = open(filename,'rb')
    full_df.append(pickle.load(file))
    file.close()


# In[ ]:


umls2020AB_df = full_df[0]


# In[ ]:


for df in full_df[1:]:
    nn_columns = df.filter(regex='.*NN.*').columns
    for col in nn_columns:
        umls2020AB_df[col] = df[col]


# In[ ]:


for col in umls2020AB_df.filter(regex='lexlm.*').columns:
    umls2020AB_df.drop(col,axis=1)


# In[ ]:


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


# In[ ]:


aui_info = []

with open('/data/Bodenreider_UMLS_DL/UMLS_VERSIONS/2020AB-ACTIVE/META/MRCONSO.RRF','r') as fp:
    
    for line in fp.readlines():
        line = line.split('|')
        cui = line[0]
        aui = line[7]
        string = line[-5]
        scui = line[9]
        source = line[11]
        
        aui_info.append({'AUI':aui, 'CUI':cui, 'STR':string, 'SCUI':scui+'|||'+source, 'SOURCE':source})
        
cui2sg = {}

with open('/data/Bodenreider_UMLS_DL/UMLS_VERSIONS/2020AB-ACTIVE/META/MRSTY.RRF','r') as fp:
    
    for line in fp.readlines():
        line = line.split('|')
        cui = line[0]
        sg = line[3]
        cui2sg[cui] = sg
        
original_umls = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/INTERSECT_AUI2ID.PICKLE','rb'))
new_auis = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/UNIQUE_AUI2ID.PICKLE','rb'))

# aui_vecs  = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/AUI2LAYER.PICKLE','rb'))
all_2020_auis = set(original_umls.keys()).union(new_auis.keys())


# In[ ]:


cui2aui = {}
aui2cui = {}
aui2scui = {}

aui2str = {}
str2aui = {}
aui2sg = {}
scui2auis = {}

cui_sg = []
cui_aui = []

for tup in tqdm(aui_info):
    current_time = time.time()
    
    aui = tup['AUI']
    scui = tup['SCUI']
    
    auis = scui2auis.get(scui, [])
    auis.append(aui)
    scui2auis[scui] = auis
    
    aui2scui[aui] = scui

    if aui in all_2020_auis:
        cui = tup['CUI']
        scui = tup['SCUI']
        string = tup['STR']
        sg = cui2sg[cui]

        auis = cui2aui.get(cui, [])
        auis.append(aui)
        cui2aui[cui] = auis
        
        aui2cui[aui] = cui
        aui2str[aui] = string
        aui2sg[aui] = sg

        auis = str2aui.get(string, [])
        auis.append(aui)
        str2aui[string] = auis
        
        cui_sg.append((cui, sg))
        cui_aui.append((cui, aui))
        
        if (time.time() - current_time) > 5:
            print(tup)
        
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


source_syn_candidates = []
source_syn_plus = []

for aui in tqdm(umls2020AB_df.auis):
    scui = aui2scui[aui]
    
    if scui.split('|||')[0] != '':       
        #Only get terms with source synonymy
        source_syns = list(set(scui2auis[scui]))
        if len(source_syns) > 500:
            break
            
        #For each source synonym, get all its 2020AA defined synonyms and add them to the candidate list
        all_syns = copy.deepcopy(source_syns)
        for source_syn_aui in source_syns:
            if source_syn_aui in aui2cui:
                AA_syns = cui2aui[aui2cui[source_syn_aui]]
                all_syns.extend(AA_syns)
    else:
        source_syns = []
        all_syns = []
        
    source_syn_candidates.append(source_syns)
    source_syn_plus.append(all_syns)
    
umls2020AB_df['source_syns'] = source_syn_candidates
umls2020AB_df['source_syns_plus'] = source_syn_plus


# ## Evaluating kNN Models + Source Synonymy 

# In[ ]:


aui_columns = umls2020AB_df.filter(regex='.*_auis').columns

query_synonym_auis = list(umls2020AB_df['2020AA_synonyms'])
source_syns = umls2020AB_df['source_syns']

for aui_col in aui_columns:
    print(aui_col)
    aui_name = aui_col.split('_auis')[0]
    nearest_neighbors_auis = umls2020AB_df[aui_col]

    #Calculating Recall @ 1,5,10,50,100
    recall_array = []

    for true_syn, top100, source in tqdm(zip(query_synonym_auis, nearest_neighbors_auis, source_syns)):

        true_syn = set(true_syn)

        source = copy.deepcopy(list(set(source)))

        if source is not None:
            source_syn_num = len(source)        
            source.extend(top100)
        else:
            source = top100        
            source_syn_num = 0

        if len(true_syn) > 0:
            recalls = []

            for n in [0,1,5,10,50,100,200,500,1000,2000]:

                topn = set(source[:n+source_syn_num])
                true_pos = topn.intersection(true_syn)

                recalls.append(len(true_pos)/len(true_syn))

            recall_array.append(recalls)
        else:
            recalls = []

            recall_array.append(recalls)

    umls2020AB_df['{}_source_syn_recall'.format(aui_name)] = recall_array



query_synonym_auis = list(umls2020AB_df['2020AA_synonyms'])
source_syns = umls2020AB_df['source_syns_plus']

for aui_col in aui_columns:
    print(aui_col)

    aui_name = aui_col.split('_auis')[0]
    nearest_neighbors_auis = umls2020AB_df[aui_col]

    #Calculating Recall @ 1,5,10,50,100
    recall_array = []

    for true_syn, top100, source in tqdm(zip(query_synonym_auis, nearest_neighbors_auis, source_syns),total=len(query_synonym_auis)):

        true_syn = set(true_syn)

        source = copy.deepcopy(list(set(source)))

        if source is not None:
            source_syn_num = len(source)        
            source.extend(top100)
        else:
            source = top100        
            source_syn_num = 0

        if len(true_syn) > 0:
            recalls = []

            for n in [0,1,5,10,50,100,200,500,1000,2000]:

                topn = set(source[:n+source_syn_num])
                true_pos = topn.intersection(true_syn)

                recalls.append(len(true_pos)/len(true_syn))

            recall_array.append(recalls)
        else:
            recalls = []

            recall_array.append(recalls)

    umls2020AB_df['{}_source_syn_plus_recall'.format(aui_name)] = recall_array


# In[ ]:


recall_df = []
names = []

for recall_col in umls2020AB_df.filter(regex='.*recall.*').columns:
    names.append(recall_col)
    recall_array = list(umls2020AB_df[recall_col].values)
    recall_df.append(pd.DataFrame(recall_array).agg(['mean']))
    

recall_df = pd.concat(recall_df)
recall_df['model'] = names

recall_df


# ## RW-UVA Using CUIs

# In[ ]:


query_synonym_auis = list(umls2020AB_df['2020AA_synonyms'])
umls2020AB_df['2020AA_synonyms_cuis'] = [[aui2cui[aui] for aui in true_syn] for true_syn in tqdm(query_synonym_auis)]


# In[ ]:


source_syns = umls2020AB_df['source_syns']

source_syn_cuis = []

for source_syn_row in tqdm(source_syns):
    
    source_syn_row_2020AA = []
    
    for source_syn_aui in source_syn_row:
        if source_syn_aui in aui2cui:
            source_syn_row_2020AA.append(aui2cui[source_syn_aui])
    
    source_syn_cuis.append(source_syn_row_2020AA)
    
umls2020AB_df['source_syns_cuis'] = source_syn_cuis


# In[ ]:


for aui_col in aui_columns:
    aui_name = aui_col.split('_auis')[0]
    nearest_neighbors_auis = umls2020AB_df[aui_col]
    umls2020AB_df['{}_cuis'.format(aui_name)] = [[aui2cui[aui] for aui in pred_syn] for pred_syn in tqdm(nearest_neighbors_auis)]


# In[ ]:


cui_columns = umls2020AB_df.filter(regex='.*2000.*_cuis').columns


# In[ ]:


cui_columns


# In[ ]:


query_synonym_cuis = list(umls2020AB_df['2020AA_synonyms_cuis'])

for cui_col in cui_columns:
    print(cui_col)
    cui_name = cui_col.split('_cuis')[0]
    nearest_neighbors_cuis = umls2020AB_df[cui_col]

    #Calculating Recall @ 1,5,10,50,100
    recall_array = []

    for true_syn, top100 in tqdm(zip(query_synonym_cuis, nearest_neighbors_cuis), total=len(query_synonym_cuis)):

        true_syn = set(true_syn)

        if len(true_syn) > 0:
            recalls = []

            for n in [1,5,10,50,100,200,500,1000,2000]:

                topn = set(top100[:n])
                true_pos = topn.intersection(true_syn)

                recalls.append(len(true_pos)/len(true_syn))

            recall_array.append(recalls)
        else:
            recalls = []

            recall_array.append(recalls)

    umls2020AB_df['{}_cui_recall'.format(cui_name)] = recall_array


# In[ ]:


query_synonym_cuis = list(umls2020AB_df['2020AA_synonyms_cuis'])
source_syns = umls2020AB_df['source_syns_cuis']

for cui_col in cui_columns:
    print(cui_col)
    cui_name = cui_col.split('_cuis')[0]
    nearest_neighbors_cuis = umls2020AB_df[cui_col]
    
    #Calculating Recall @ 1,5,10,50,100
    recall_array = []

    for true_syn, top100, source in tqdm(zip(query_synonym_cuis, nearest_neighbors_cuis, source_syns), total=len(query_synonym_cuis)):

        true_syn = set(true_syn)
        source = copy.deepcopy(list(set(source)))

        if len(true_syn) > 0:
            recalls = []

            if source is not None:
                source_syn_num = len(source)        
                source.extend(top100)
            else:
                source = top100
                source_syn_num = 0

            for n in [0,1,5,10,50,100,200,500,1000,2000]:
                topn = set(source[:n+source_syn_num])
                true_pos = topn.intersection(true_syn)

                recalls.append(len(true_pos)/len(true_syn))

            recall_array.append(recalls)
        else:
            recalls = []

            recall_array.append(recalls)

    umls2020AB_df['{}_source_syn_cui_recall'.format(cui_name)] = recall_array


# In[ ]:


umls2020AB_df['number_source_syn_cuis'] = [len(set(c)) for c in umls2020AB_df.source_syns_cuis] 
umls2020AB_df['number_source_syn_auis'] = [len(set(c)) for c in umls2020AB_df.source_syns] 
umls2020AB_df['number_source_syn_plus_auis'] = [len(set(c)) for c in umls2020AB_df.source_syns_plus] 


# In[ ]:


recall_df = []
names = []

for recall_col in umls2020AB_df.filter(regex='.*recall.*').columns:
    names.append(recall_col)
    recall_array = list(umls2020AB_df[recall_col].values)
    recall_df.append(pd.DataFrame(recall_array).agg(['mean']))

recall_df = pd.concat(recall_df)
recall_df['model'] = names

recall_df.to_csv('recall_cui_knn_source.csv')