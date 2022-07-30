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


class UMLS():
    
    def __init__(self):
        
        self.aui_info = []
        self.cui2sg = {}
        
        self.cui2aui = {}
        self.aui2cui = {}
        self.aui2scui = {}
        self.cui2preferred = {}

        self.aui2str = {}
        self.str2aui = {}
        self.aui2sg = {}
        self.scui2auis = {}

        self.cui_sg = []
        self.cui_aui = []

        self.semtype2sg = {}

    def load_umls(self, version='2020AB'):

        with open('/data/Bodenreider_UMLS_DL/UMLS_VERSIONS/{}-ACTIVE/META/MRCONSO.RRF'.format(version),'r') as fp:

            pbar = tqdm(total=15000000)
            line = fp.readline()

            while line:
                line = line.split('|')
                cui = line[0]
                aui = line[7]
                string = line[-5]
                scui = line[9]
                source = line[11]
                term_status = line[2]
                string_type = line[4]
                language = line[1]

                self.aui_info.append((aui,cui,string,scui,source, term_status, string_type, language))

                line = fp.readline()
                pbar.update(1)


        with open('/data/Bodenreider_UMLS_DL/UMLS_VERSIONS/{}-ACTIVE/META/MRSTY.RRF'.format(version),'r') as fp:

            for line in fp.readlines():
                line = line.split('|')
                cui = line[0]
                sg = line[3]
                self.cui2sg[cui] = sg

        original_umls = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/INTERSECT_AUI2ID.PICKLE','rb'))
        new_auis = pickle.load(open('/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/UNIQUE_AUI2ID.PICKLE','rb'))

        all_2020_auis = set(original_umls.keys()).union(new_auis.keys())

        for tup in tqdm(self.aui_info):

            tup = {'AUI':tup[0], 
                   'CUI':tup[1], 
                   'STR':tup[2], 
                   'SCUI':tup[3]+'|||'+tup[4], 
                   'SOURCE':tup[4], 
                   'PREF':(tup[5], tup[6]), 
                   'LANG':tup[7]}

            current_time = time.time()

            aui = tup['AUI']
            scui = tup['SCUI']

            auis = self.scui2auis.get(scui, [])
            auis.append(aui)
            self.scui2auis[scui] = auis

            self.aui2scui[aui] = scui

            cui = tup['CUI']
            string = tup['STR']

            pref = tup['PREF']

            if pref[0] == 'P' and pref[1] == 'PF' and tup['LANG'] == 'ENG':
                self.cui2preferred[cui] = string

            self.aui2str[aui] = string
            self.aui2cui[aui] = cui
            self.aui2sg[aui] = sg

            auis = self.str2aui.get(string, [])
            auis.append(aui)
            self.str2aui[string] = auis

            self.cui_sg.append((cui, sg))
            self.cui_aui.append((cui, aui))

            if aui in all_2020_auis:
                scui = tup['SCUI']
                sg = self.cui2sg[cui]

                auis = self.cui2aui.get(cui, [])
                auis.append(aui)
                self.cui2aui[cui] = auis

                if (time.time() - current_time) > 5:
                    print(tup)

        self.semgroups = pd.read_csv('SemGroups.txt',sep='|',header=None)

        for i, row in self.semgroups.iterrows():

            st = row[3]
            sg = row[1]

            self.semtype2sg[st] = sg