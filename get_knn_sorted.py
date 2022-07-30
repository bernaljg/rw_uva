#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import copy
import faiss
import gc
import subprocess

import sys

print('Loading Strings')

sorted_umls_df = pd.read_csv('/data/Bodenreider_UMLS_DL/Interns/Bernal/sorted_umls2020_auis.csv',sep='\t',index_col=0)
sorted_umls_df = sorted_umls_df.sort_values('0',ascending=False)


# In[9]:


original_umls_2020, new_umls_2020 = pickle.load(open('aui_string_map_UMLS2020_update.p','rb'))

original_auis = set([x[0] for x in original_umls_2020])


# In[ ]:


synonym_dict = pickle.load(open('new_umls_synonym_aui_dict.p','rb'))


# In[ ]:


new = []
synonym_list = []

for aui in tqdm(sorted_umls_df.auis):
    
    if aui in original_auis:
        new.append(False)
        synonym_list.append(None)
    else:
        new.append(True)
        synonyms = synonym_dict[aui]
        new_synonyms = []
        
        for aui in synonyms:
            if aui in original_auis:
                new_synonyms.append(aui)
                
        synonym_list.append(new_synonyms)


# In[ ]:


sorted_umls_df['2020AB?'] = new
sorted_umls_df['2020AA_synonyms'] = synonym_list


# In[ ]:


sorted_umls_df.groupby('2020AB?').count()


# In[10]:


print('Loading Vectors')

model_name = sys.argv[1]
vectors_name = '{}_vecs'.format(model_name)


vecs = []
for i in range(167):
    vecs.append(pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}/umls2020_{}_{}.p'.format(vectors_name, vectors_name, i),'rb')))
    
vecs = np.vstack(vecs)


# In[ ]:


sorted_umls_df[vectors_name] = list(vecs)


# In[ ]:


umls2020AA_df = sorted_umls_df[sorted_umls_df['2020AB?'] == False][['0','strings','auis']]
umls2020AA_vecs = sorted_umls_df[sorted_umls_df['2020AB?'] == False][vectors_name]
umls2020AA_vecs = np.vstack(umls2020AA_vecs)


# In[ ]:


umls2020AB_df = sorted_umls_df[sorted_umls_df['2020AB?']][['0','strings','auis','2020AA_synonyms']]
umls2020AB_vecs = sorted_umls_df[sorted_umls_df['2020AB?']][vectors_name]
umls2020AB_vecs = np.vstack(umls2020AB_vecs)


# In[ ]:
print('Chunking')


dim = len(vecs[0])
index_split = 3
index_chunks = np.array_split(umls2020AA_vecs,index_split)
query_chunks = np.array_split(umls2020AB_vecs,100)

k = 2000

print('Building Index')

index_chunk_D = []
index_chunk_I = []

current_zero_index = 0

for num, index_chunk in enumerate(index_chunks):
    
    print('Running Index Part {}'.format(num))
    
    index = faiss.IndexFlatL2(dim)   # build the index
        
    if faiss.get_num_gpus() > 1:
        gpu_resources = []

        for i in range(faiss.get_num_gpus()):
            res = faiss.StandardGpuResources()
            gpu_resources.append(res)

        gpu_index = faiss.index_cpu_to_gpu_multiple_py(gpu_resources, index)
    else:
        gpu_resources = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
    
    print()
    gpu_index.add(index_chunk)

    D, I = [],[]

    for q in tqdm(query_chunks):
        d,i = gpu_index.search(q, k)

        i += current_zero_index
        
        D.append(d)
        I.append(i)
        
    index_chunk_D.append(D)
    index_chunk_I.append(I)
    
    current_zero_index += len(index_chunk)
    
    print(subprocess.check_output(['nvidia-smi']))

    del gpu_index
    del gpu_resources
    gc.collect()


print('Combining Index Chunks')
          
stacked_D = []
stacked_I = []

for D,I in zip(index_chunk_D, index_chunk_I):
    
    D = np.vstack(D)
    I = np.vstack(I)
    
    stacked_D.append(D)
    stacked_I.append(I)


# In[ ]:

del index_chunk_D
del index_chunk_I
gc.collect()

stacked_D = np.hstack(stacked_D)
stacked_I = np.hstack(stacked_I)


# In[ ]:


full_sort_I = []
full_sort_D = []

for d, i in tqdm(zip(stacked_D, stacked_I)):
    
    sort_indices = np.argsort(d)
    
    i = i[sort_indices][:k]
    d = d[sort_indices][:k]
    
    full_sort_I.append(i)
    full_sort_D.append(d)

del stacked_D
del stacked_I
gc.collect()

umls_2020AA_auis = list(umls2020AA_df.auis)


# In[ ]:


nearest_neighbors_auis = []

for nn_inds in tqdm(full_sort_I):
    
    nn_auis = [umls_2020AA_auis[i] for i in nn_inds]
    
    nearest_neighbors_auis.append(nn_auis)


# In[ ]:


query_synonym_auis = list(umls2020AB_df['2020AA_synonyms'])


# In[ ]:


#Calculating Recall @ 1,5,10,50,100
recall_array = []
closest_dist_true = []
closest_dist_false = []

for true_syn, top100, top100_dist in tqdm(zip(query_synonym_auis, nearest_neighbors_auis, full_sort_D)):
    
    true_syn = set(true_syn)
    
    if len(true_syn) > 0:
        recalls = []

        for n in [1,5,10,50,100,200,500,1000,2000]:

            topn = set(top100[:n])
            true_pos = topn.intersection(true_syn)

            recalls.append(len(true_pos)/len(true_syn))

        recall_array.append(recalls)
        closest_dist_true.append([top100_dist[0], np.mean(top100_dist)])
    else:
        recalls = []

        recall_array.append(recalls)
        closest_dist_false.append([top100_dist[0], np.mean(top100_dist)])


# In[ ]:


umls2020AA_aui2str = {}

for aui, string in tqdm(zip(umls2020AA_df.auis, umls2020AA_df.strings)):
    umls2020AA_aui2str[aui] = string


# In[ ]:


nearest_neighbors_strings = []

for nn_auis in tqdm(nearest_neighbors_auis):
    nn_strings = [umls2020AA_aui2str[aui] for aui in nn_auis]
    
    nearest_neighbors_strings.append(nn_strings)


# In[ ]:


synonym_strings = []

for syn_auis in tqdm(umls2020AB_df['2020AA_synonyms']):
    syn_strings = [umls2020AA_aui2str[aui] for aui in syn_auis]
    
    synonym_strings.append(syn_strings)


# In[ ]:


umls2020AB_df['synonym_strings'] = synonym_strings


# In[ ]:


umls2020AB_df['num_syms'] = [len(s) for s in umls2020AB_df['2020AA_synonyms']]


# In[ ]:


umls2020AB_df['{}_{}-NN_strings'.format(model_name, k)] = nearest_neighbors_strings
umls2020AB_df['{}_{}-NN_auis'.format(model_name, k)] = nearest_neighbors_auis
umls2020AB_df['{}_{}-NN_dist'.format(model_name, k)] = list(full_sort_D)


# In[ ]:


umls2020AB_df['{}_{}-NN_recall'.format(model_name, k)] = recall_array


print('Dumping File')

pickle.dump(umls2020AB_df, open('/data/Bodenreider_UMLS_DL/Interns/Bernal/UMLS2020AB_{}.{}-NN_DataFrame.p'.format(model_name, k),'wb'))

