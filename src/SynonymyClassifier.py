import _pickle as pickle
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
from glob import glob
import json
from utils import *
import os

import ipdb

class SynonymyDatasetManager:

    def __init__(self,
                 umls,
                 retrieval_pipeline,
                 output_dir,
                 num_candidates,
                 dev_perc,
                 test_perc,
                 add_gold_candidates
                 ):

        self.umls = umls
        self.retrieval_pipeline = retrieval_pipeline

        self.num_candidates = num_candidates

        self.dev_perc = dev_perc
        self.test_perc = test_perc

        self.add_gold_candidates = add_gold_candidates

        self.datasets = {}
        self.retrieved_candidates_by_split = None

        self.dataset_generation_done = False

        output_dir = output_dir + '/classifiers'

        if not(os.path.exists(output_dir)):
            os.makedirs(output_dir)

        # Define output directory as data directory
        self.output_dir = output_dir

        # Make Unique Directory for this retrieval procedure
        configs = {'Number of Candidates to Classify': num_candidates,
                   'Dev Percentage': dev_perc,
                   'Test Percentage': test_perc,
                   'Add Gold Candidates': add_gold_candidates
                   }
        directories = glob(output_dir + '/*')

        new_directory_num = len(directories)

        for dir in directories:
            prev_config = json.load(open('{}/config.json'.format(dir),'r'))

            if equivalent_dict(prev_config, configs):
                if os.path.exists('{}/dataset_generation_done.json'.format(dir)):
                    print('Configuration Already Done and Saved.')
                    self.dataset_generation_done = True
                else:
                    print('Previous Run Stopped. Running Again.')

                new_directory_num = dir.split('/')[-1]

        self.output_dir = output_dir + '/{}'.format(new_directory_num)

        if not(os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)
            json.dump(configs, open(self.output_dir + '/config.json', 'w'))

    def create_classification_dataset(self):
        """
        This method uses the set of new AUIs to create a pairwise synonymy identification dataset.
        After the retrieval pipeline has been run, the new AUIs are split into train/dev/test
        subsets based on concept IDs or CUIs (each split a predetermined percentage of total data).

        It then saves all three dataset splits within the same output directory.

        Args:
            dev_perc: Percentage or number of total concepts to add to dev subset (% or # of CUIs, not synonym pairs)
            test_perc: Percentage or number of total concepts to add to test subset (% or # of CUIs, not synonym pairs)
            add_gold_candidates: Whether dataset will contain all gold annotated synonyms
            even if retrieval missed them
        """

        if self.dataset_generation_done:
            print('Loading Dataset.')
            self.aui_splits = pickle.load(open('{}/aui_splits.p'.format(self.output_dir), 'rb'))
        else:
            self.create_new_aui_info_df()
            self.split_dataset()
            self.generate_all_synonym_pairs()
            self.save_pickle_and_csv_datasets()

            json.dump({'DONE': True}, open(self.output_dir + '/dataset_generation_done.json', 'w'))


    def create_new_aui_info_df(self):
        new_cuis_plus_info = []

        for aui in self.retrieval_pipeline.new_auis:
            string = self.umls.aui2str[aui]
            cui = self.umls.aui2cui[aui]
            sg = self.umls.aui2sg[aui]
            original_syns = self.umls.original_only_cui2auis.get(cui, [])
            num_original_syns = len(original_syns)

            new_cuis_plus_info.append((aui, string, cui, sg, original_syns, num_original_syns))

        self.new_cuis_df = pd.DataFrame(new_cuis_plus_info, columns=['aui', 'string', 'cui', 'sg', 'original_syns', 'num_original_syns'])

    def split_dataset(self, stratified_method='synonym_presence'):
        """

        Args:
            stratified_method: Method for dataset split stratification
                - synonym_presence: Split by whether original ontology has synonyms for said AUI or not
                - semantic_group: Split by semantic groups
        """

        train, dev, test = self.get_cui_split(stratified_method)

        # Classify each new AUI into each split
        split = []

        for cui in self.new_cuis_df['cui']:
            if cui in train:
                split.append('train')
            elif cui in dev:
                split.append('dev')
            elif cui in test:
                split.append('test')

        self.new_cuis_df['split'] = split

    def get_cui_split(self, stratified_method):
        train, dev, test = set(), set(), set()

        if stratified_method == 'synonym_presence':

            #Splitting Based on CUIs
            cui_num_syms_df = self.new_cuis_df[['cui', 'num_original_syns']].drop_duplicates()
            cui_num_syms_df['no_syms'] = [n == 0 for n in cui_num_syms_df['num_original_syns']]

            train = []
            dev = []
            test = []

            #Converting to a percentage if number of samples used for splitting
            if self.dev_perc > 1:
                self.dev_perc /= len(cui_num_syms_df)

            if self.test_perc > 1:
                self.test_perc /= len(cui_num_syms_df)

            ipdb.set_trace()

            for i, g in cui_num_syms_df.groupby('no_syms'):
                perm_g = g.sample(len(g), random_state=np.random.RandomState(42))['cui'].values

                train.extend(perm_g[:len(g) - int(len(g) * (self.dev_perc + self.test_perc))])
                dev.extend(perm_g[len(g) - int(len(g) * (self.dev_perc + self.test_perc)):len(g) - int(len(g) * (self.test_perc))])
                test.extend(perm_g[len(g) - int(len(g) * self.test_perc):])

                assert (train[-1] != dev[0])
                assert (dev[-1] != test[0])

            train = set(train)
            dev = set(dev)
            test = set(test)

        return train, dev, test

    def generate_all_synonym_pairs(self):
        # Remove AUIs which share strings and link to the same CUIs
        dedup_df = []

        for i, g in tqdm(self.new_cuis_df.groupby(['string', 'cui'])):

            for j, row in g.iterrows():
                dedup_df.append(row)
                break

        dedup_df = pd.DataFrame(dedup_df)

        #Create Split Dictionary
        self.aui_splits = {}

        for i, row in tqdm(dedup_df.iterrows(), total=len(dedup_df)):
            split = row['split']
            aui = row['aui']
            syns = row['original_syns']

            aui_samples = self.aui_splits.get(split, set())

            candidate_auis = self.retrieval_pipeline.retrieved_candidates[aui][:self.num_candidates]

            if self.add_gold_candidates:
                for syn in syns:
                    aui_samples.add((aui, syn, 1))

            for aui_cand in candidate_auis:
                if aui_cand in syns:
                    label = 1
                else:
                    label = 0

                aui_sample = (aui, aui_cand, label)

                aui_samples.append(aui_sample)

            self.aui_splits[split] = set(aui_samples)

    def save_pickle_and_csv_datasets(self):

        # Saving pairs
        pickle.dump(self.aui_splits, open('{}/aui_splits.p'.format(self.output_dir),'wb'))

        #
        for split, tups in self.aui_splits.items():
            
            one_way = []
            two_way = []

            for aui1, aui2, label in tqdm(tups):

                str1 = self.umls.aui2str[aui1]
                str2 = self.umls.aui2str[aui2]

                one_way.append((str1 + ' [SEP] ' + str2, label))

                if split == 'train':
                    two_way.append((str1 + ' [SEP] ' + str2, label))
                    two_way.append((str2 + ' [SEP] ' + str1, label))

            one_way_df = pd.DataFrame(one_way, columns=['sents', 'labels'])
            one_way_df = one_way_df.sample(len(one_way_df), random_state=np.random.RandomState(42))

            if split == 'train':
                two_way_df = pd.DataFrame(two_way, columns=['sents', 'labels'])
                two_way_df = two_way_df.sample(len(two_way_df), random_state=np.random.RandomState(42))
            else:
                two_way_df = one_way_df

            one_way_df.to_csv('{}/one-way.{}.tsv'.format(self.output_dir, split), sep='\t', quoting=3)
            two_way_df.to_csv('{}/two-way.{}.tsv'.format(self.output_dir, split), sep='\t', quoting=3)

class SynonymyClassifier:
    pass
