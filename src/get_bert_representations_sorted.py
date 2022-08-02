import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

import sys


pt_model = sys.argv[1]
model = AutoModel.from_pretrained(pt_model)
model.to('cuda')

tokenizer = AutoTokenizer.from_pretrained(pt_model)

print('Loading Strings')

sorted_umls_df = pd.read_csv('/data/Bodenreider_UMLS_DL/Interns/Bernal/sorted_umls2020_auis.csv',sep='\t',index_col=0)

print('Start Encoding')

sorted_umls_df = sorted_umls_df.sort_values('0',ascending=False)

all_cls = []

if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_vecs'.format(sys.argv[2]))):
    os.makedirs('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_vecs'.format(sys.argv[2]))
               
with torch.no_grad():
    
    num_strings_proc = 0
    vec_save_batch_num = 0    
    batch_sizes = []
    
    text_batch = []
    pad_size = 0
    
    curr_vecs = 0
    
    for i,row in tqdm(sorted_umls_df.iterrows(),total=len(sorted_umls_df)):
        
        string = str(row['strings'])
        length = row[0]
        
        text_batch.append(string)
        num_strings_proc += 1
        
        if length > pad_size:
            pad_size = length
        
        if pad_size * len(text_batch) > 6000 or num_strings_proc == len(sorted_umls_df):

            if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_vecs/umls2020_{}_vecs_{}.p'.format(sys.argv[2],sys.argv[2],vec_save_batch_num))):
                text_batch = list(text_batch)
                encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,max_length=model.config.max_length)
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']

                input_ids = input_ids.to('cuda')
                attention_mask = attention_mask.to('cuda')

                outputs = model(input_ids, attention_mask=attention_mask)
                all_cls.append(outputs[0][:,0,:].cpu().numpy())
            
            batch_sizes.append(len(text_batch))
            curr_vecs += 1
            
            text_batch = []
            pad_size = 0
            
            if curr_vecs == 100:
                print('Latest_batch_size {}'.format(batch_sizes[-1]))
                print(sum(batch_sizes))
                if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_vecs/umls2020_{}_vecs_{}.p'.format(sys.argv[2],sys.argv[2],vec_save_batch_num))):
                    all_cls = np.vstack(all_cls)
                    pickle.dump(all_cls, open('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_vecs/umls2020_{}_vecs_{}.p'.format(sys.argv[2],sys.argv[2],vec_save_batch_num),'wb'))
                
                vec_save_batch_num += 1
                all_cls = []
                curr_vecs = 0
                
    if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_vecs/umls2020_{}_vecs_{}.p'.format(sys.argv[2],sys.argv[2],vec_save_batch_num))):
        all_cls = np.vstack(all_cls)
        pickle.dump(all_cls, open('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_vecs/umls2020_{}_vecs_{}.p'.format(sys.argv[2],sys.argv[2],vec_save_batch_num),'wb'))