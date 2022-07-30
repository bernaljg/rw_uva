import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
 
index = {}
 
with open('/data/Bodenmentions_per_abstract.csv','r') as f:
 
    line = f.readline()
 
    doc_ind = 0
 
    while line is not None:
        line = line.strip()
 
        if line == '':
            if doc_ind % 100000 == 0:
                print(doc_ind)
 
            #Save Doc in Co-Occurence Dictionary            
            doc_ind += 1
 
        term_docs = index.get(line, [])
        term_docs.append(doc_ind)
        index[line] = term_docs
 
        line = f.readline()
 
pickle.dump(index,open('pubmed_term_index.p','wb'))
