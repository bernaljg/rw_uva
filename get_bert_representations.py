#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from tqdm import tqdm


# In[2]:


import torch
import transformers
from transformers import AutoModel, AutoTokenizer


# In[3]:


pt_model = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
model = AutoModel.from_pretrained(pt_model)
model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(pt_model)


# In[4]:

print('Loading Strings')

original_umls_2020, new_umls_2020 = pickle.load(open('aui_string_map_UMLS2020_update.p','rb'))


# In[6]:


aui_strings = [t[1] for t in original_umls_2020]
aui_strings.extend([t[1] for t in new_umls_2020])


# In[7]:

# aui_strings = aui_strings[:100000]

print('Creating Batches')
bins = int(len(aui_strings)/100)


# In[8]:

batches = np.array_split(aui_strings, bins)


# In[9]:

print('Start Encoding')

all_cls = []

with torch.no_grad():
    
    i = 0
    num_proc = 0

    for text_batch in tqdm(batches):
        if os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/umls2020_sapbert_vecs_{}.p'.format(i)):
            text_batch = list(text_batch)
            encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=False)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')

            outputs = model(input_ids, attention_mask=attention_mask)
            all_cls.append(outputs[0][:,0,:].cpu().numpy())

        num_proc += 1

        if num_proc % 100 == 0:
            if len(all_cls) == 100:
                all_cls = np.vstack(all_cls)
            
                pickle.dump(all_cls, open('/data/Bodenreider_UMLS_DL/Interns/Bernal/umls2020_sapbert_vecs_{}.p'.format(i),'wb'))
            
            i += 1
            all_cls = []
