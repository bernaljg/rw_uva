import _pickle as pickle
import json

import pandas as pd
from tqdm import tqdm
import time
from UMLS import *
from RetrievalModule import *

from glob import glob


def equivalent_dict(prev_config, configs):
    for key in prev_config.keys():
        if prev_config[key] != configs[key]:
            return False

    return True


class RetrievalPipeline:
    """
    Class designed to run several retrieval modules.
    """

    def __init__(self,
                 original_auis_filename,
                 new_auis_filename,
                 ontology,
                 output_dir,
                 maximum_candidates_per_retriever,
                 retriever_names
                 ):

        self.ontology = ontology

        # Rebuild mappings using original and new aui list for this UMLS augmentation experiment
        self.original_auis, self.new_auis = self.ontology.get_relevant_aui_set(original_auis_filename,
                                                                               new_auis_filename)

        # Load Original and New AUIs (Synonyms obtained only from AUIs in these 2 sets)
        self.ontology.reset_mappings()
        self.ontology.create_mappings()

        self.maximum_candidates_per_retriever = maximum_candidates_per_retriever
        self.retriever_names = retriever_names

        self.retrieved_candidates = {}

        self.relevant_auis = self.ontology.relevant_auis

        # Make Unique Directory for this retrieval procedure
        configs = {'UMLS Version': ontology.version,
                   'UMLS Directory': ontology.directory,
                   'Retriever Names': retriever_names,
                   'Original AUI Filename': original_auis_filename,
                   'New AUI Filename': new_auis_filename,
                   'Maximum Candidates per Retriever': maximum_candidates_per_retriever}
        retrieval_directories = glob(output_dir + '/*')

        for dir in retrieval_directories:
            prev_config = json.load(dir)

            if equivalent_dict(prev_config, configs):
                assert print('Configuration Already Done and Saved')

        self.output_dir = output_dir + '/{}'.format(len(retrieval_directories))
        os.makedirs(self.output_dir)

        json.dump(configs, self.output_dir + '/config.json')

    def load_retrievers(self):
        self.retrievers = []

        for retriever_name in self.retriever_names:
            self.retrievers.append(RetrievalModule(retriever_name,
                                                   self))

    def combine_candidates(self, new_candidate_dict, add_on_top=False):

        for new_aui in self.new_auis:

            current_candidates = self.retrieved_candidates.get(new_aui, [])
            new_candidates = new_candidate_dict[new_aui]

            # Add new candidates before or after previous ones
            if add_on_top:
                current_candidates = new_candidates + current_candidates
            else:
                current_candidates = current_candidates + new_candidates

            self.retrieved_candidates[new_aui] = current_candidates

    def run_retrievers(self,
                       exclude=[]):

        for ret_name, ret in zip(self.retriever_names, self.retrievers):

            if ret_name not in exclude:
                new_candidate_dict = ret.retrieve()
                self.eval_and_save_candidates(new_candidate_dict, ret_name)
                self.combine_candidates(new_candidate_dict, ret.add_on_top)

        self.eval_and_save_candidates(self.retrieved_candidates, 'full_pipeline')

    def evaluate_candidate_retrieval(self,
                                     mode,
                                     recall_at=[1, 5, 10, 50, 100, 200, 500, 1000, 2000]):

        if self.ontology.original_only_cui2auis is None:
            print('Populating original only synonyms before evaluation.')
            self.ontology.get_original_ontology_synonyms()

        new_auis = []
        recall_array = []

        for new_aui, candidates in tqdm(self.retrieved_candidates.items()):
            new_auis.append(new_aui)

            cui = self.ontology.aui2cui[new_aui]
            true_syn = set(self.ontology.original_only_cui2auis.get(cui, []))

            if len(true_syn) > 0:
                if mode == 'CUI':
                    true_syn = {cui}
                    candidates = [self.ontology.aui2cui[aui] for aui in candidates]

                recalls = []

                for n in recall_at:
                    topn = set(candidates[:n])
                    true_pos = topn.intersection(true_syn)

                    # Number of true positives in first n over the number of possible positive candidates
                    # (if n is less than the number of true synonyms it is impossible to correctly recall all of them)
                    recall_at_n = len(true_pos) / min(len(true_syn), n)
                    recalls.append(recall_at_n)

                recall_array.append(recalls)
            else:
                recalls = []

                recall_array.append(recalls)

        return pd.DataFrame(recall_array, index=new_auis, columns=['R@{}'.format(n) for n in recall_at])

    def eval_and_save_candidates(self, candidate_dict, ret_name):
        ret_name = ret_name.replace('/', '_')  # In case using filename

        aui_recall = self.evaluate_candidate_retrieval(mode='AUI')
        cui_recall = self.evaluate_candidate_retrieval(mode='CUI')

        aui_recall.to_csv('{}/{}_aui_recall_complete.csv'.format(self.output_dir, ret_name))
        cui_recall.to_csv('{}/{}_cui_recall_complete.csv'.format(self.output_dir, ret_name))

        aui_mean_row = aui_recall.mean()
        cui_mean_row = cui_recall.mean()
        metrics = pd.concat([aui_mean_row, cui_mean_row])
        metrics.index = ['{}_AUI_metrics'.format(ret_name), '{}_CUI_metrics'.format(ret_name)]
        metrics.to_csv('{}/{}_recall_summary.csv'.format(self.output_dir, ret_name))

        pickle.dump(candidate_dict, open('{}/{}_candidates.p'.format(self.output_dir, ret_name)))
