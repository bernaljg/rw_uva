import _pickle as pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import copy
import glob
import gc

import xml.etree.ElementTree as ET


pubmed_xmls = glob.glob('/data/Bodenreider_UMLS_DL/Interns/Bernal/pubmed_abstracts/baseline/pubmed*.xml')

for file in tqdm(pubmed_xmls):

    abstracts_per_article = []
    dates_per_article = []
    titles = []

    tree = ET.iterparse(file)
#     root = tree.getroot()

    for event, article in tree:
        if article.tag == 'PubmedArticle':

            dates = []
            abstracts = []
            pmid = []
            title = []

            for i in article.iter():
                tag = str(i.tag)
                
                if 'AbstractText' in tag:
                    abstracts.append(i.text)

                if 'Year' in tag:
                    dates.append(i.text)

                if tag == 'ArticleTitle':
                    title.append(i.text)

            date = max([int(y) for y in dates])

            if len(abstracts) > 0:
                abstracts_per_article.append(abstracts)
                dates_per_article.append(date)
                assert len(set(title)) == 1, ipdb.set_trace()
                titles.append(title[0])
                
    direct = '/'.join(file.split('/')[:-2]) + '/raw_abstracts'
    file_id = file.split('/')[-1].split('.')[0]
    
    if not(os.path.exists(direct)):
        os.makedirs(direct)
        
    pickle.dump((titles, dates_per_article ,abstracts_per_article), open(direct + '/' + file_id + '.p','wb'))