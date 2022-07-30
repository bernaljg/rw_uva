import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch
import transformers
from transformers import AutoModel, BertTokenizer, AutoTokenizer, AutoModelForNextSentencePrediction

import sys

tokenizer_filename = sys.argv[1]
pt_model = sys.argv[2]
summary_model_name  = sys.argv[3]
edges_pickle = sys.argv[4]

if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_predictions'.format(summary_model_name))):
    os.makedirs('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_predictions'.format(summary_model_name))

try:
    tokenizer = BertTokenizer(tokenizer_filename)
except:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_filename)
    
model = AutoModelForNextSentencePrediction.from_pretrained(pt_model)
model.to('cuda')

print('Loading Strings')

real_world_pairs = pickle.load(open(edges_pickle,'rb'))

print('Start Classifying')

all_cls = []
    
with torch.no_grad():
    
    num_strings_proc = 0
    vec_save_batch_num = 0    
    batch_sizes = []
    
    text_batch = []
    pad_size = 0
    
    curr_predictions = 0
    
    for head, tail, syn, in tqdm(real_world_pairs):
        head = str(head)
        tail = str(tail)
        
        forward = head + ' [SEP] ' + tail + ' [SEP] '
        backward = tail + ' [SEP] ' + head + ' [SEP] '
                
        length = max(len(forward),len(backward))/3
        
        text_batch.append((head,tail))
        text_batch.append((tail,head))
        
        num_strings_proc += 1
        
        if length > pad_size:
            pad_size = length
        
        if pad_size * len(text_batch) > 6000 or num_strings_proc == len(real_world_pairs):

            if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_predictions/umls2020_{}_predictions_{}.p'.format(summary_model_name,summary_model_name,vec_save_batch_num))):
                
                text_batch = list(text_batch)
            
                encoding = tokenizer.batch_encode_plus(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=512)

                input_ids = encoding['input_ids']
                token_type_ids = encoding['token_type_ids']
                attention_mask = encoding['attention_mask']

                input_ids = input_ids.to('cuda')
                token_type_ids = token_type_ids.to('cuda')
                attention_mask = attention_mask.to('cuda')

                outputs = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                all_cls.append(outputs[0].cpu().numpy())
            
            batch_sizes.append(len(text_batch))
            curr_predictions += 1
            
            text_batch = []
            pad_size = 0
            
            if curr_predictions == 100:
                print('Latest_batch_size {}'.format(batch_sizes[-1]))
                print(sum(batch_sizes))
                if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_predictions/umls2020_{}_predictions_{}.p'.format(summary_model_name,summary_model_name,vec_save_batch_num))):
                    all_cls = np.vstack(all_cls)
                    pickle.dump(all_cls, open('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_predictions/umls2020_{}_predictions_{}.p'.format(summary_model_name,summary_model_name,vec_save_batch_num),'wb'))
                
                vec_save_batch_num += 1
                all_cls = []
                curr_predictions = 0
                
    if not(os.path.exists('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_predictions/umls2020_{}_predictions_{}.p'.format(summary_model_name,summary_model_name,vec_save_batch_num))):
        all_cls = np.vstack(all_cls)
        pickle.dump(all_cls, open('/data/Bodenreider_UMLS_DL/Interns/Bernal/{}_predictions/umls2020_{}_predictions_{}.p'.format(summary_model_name,summary_model_name,vec_save_batch_num),'wb'))