import _pickle as pickle
from glob import glob
import os.path
import sys

import pandas as pd

import pickle
import numpy as np
import os
from tqdm import tqdm
import torch

import copy
import faiss
import gc
import subprocess
import time

from transformers import AutoModel, AutoTokenizer

retrieval_modules_types_priority = {
    'scui_cui': True,
    'scui_lui': True,
    'scui': True
}

# TODO: Change hard-coded vector output directory
VECTOR_DIR = '/data/Bodenreider_UMLS_DL/Interns/Bernal/rw_uva/data/plm_vectors'


class RetrievalModule:
    """
    Class designed to retrieve potential synonymy candidates for a set of UMLS terms from a set of entities.
    """

    def __init__(self,
                 retriever_name,
                 retriever_pipeline):
        """
        Args:
            retriever_name: Retrieval names can be one of 3 types
                1) "scui_cui" and "scui_lui"
                2) The name of a pickle file mapping AUIs to precomputed vectors
                3) A huggingface transformer model

            retriever_pipeline:
        """

        self.add_on_top = retrieval_modules_types_priority.get(retriever_name, False)
        self.retriever_name = retriever_name

        # Contains all necessary ontology information
        self.retriever_pipeline = retriever_pipeline
        self.ontology = retriever_pipeline.ontology

        # If retriever name is not in global priority dictionary then it relies on dense retrieval.
        if self.retriever_name not in retrieval_modules_types_priority:

            self.load_or_create_sorted_aui_df()

            # Search for pickle file
            if os.path.exists(retriever_name):
                self.aui_vector_dict = pickle.load(open(retriever_name, 'rb'))
            else:
                print('No Pre-Computed Vectors. Confirming PLM Model.')

                try:
                    self.plm = AutoModel.from_pretrained(retriever_name)
                except:
                    assert False, print('Invalid Retriever Name. Check Documentation.')

                aui_vectors = self.get_plm_vectors()
                self.populate_vector_dictionary(aui_vectors)

                print('Vectors Loaded.')

    def retrieve(self):
        # If retriever name is not in global priority dictionary then it relies on dense retrieval.
        if self.retriever_name not in retrieval_modules_types_priority:
            return self.retrieve_knn()
        elif self.retriever_name == 'scui_cui':
            return self.retrieve_scui_cui()
        elif self.retriever_name == 'scui_lui':
            return self.retrieve_scui_lui()
        elif self.retriever_name == 'scui':
            return self.retrieve_scui()

    def get_plm_vectors(self):

        # Load or Create a DataFrame sorted by phrase length for efficient PLM computation
        self.load_or_create_sorted_aui_df()

        # If not pre-computed, create vectors
        retrieval_name_dir = VECTOR_DIR + '/' + self.retriever_name.replace('/', '_')

        if not (os.path.exists(retrieval_name_dir)):
            os.makedirs(retrieval_name_dir)

            print('Computing AUI Vectors. Make sure this process is equipped with at least 1 GPU.')
            self.compute_plm_vectors(retrieval_name_dir)

        return self.load_plm_vectors(retrieval_name_dir)

    def load_or_create_sorted_aui_df(self):

        if self.ontology.version == '2020AB':
            self.sorted_umls_df = pd.read_csv('/data/Bodenreider_UMLS_DL/Interns/Bernal/sorted_umls2020_auis.csv',
                                              sep='\t', index_col=0)
            self.sorted_umls_df = self.sorted_umls_df.sort_values('0', ascending=False)
        else:
            auis = list(self.retriever_pipeline.relevant_auis)
            strings = []
            lengths = []

            for aui in tqdm(auis):
                string = self.ontology.aui2str[aui]
                lengths.append(len(string))

            lengths_df = pd.DataFrame(lengths)
            lengths_df['strings'] = strings
            lengths_df['auis'] = auis

            self.sorted_umls_df = lengths_df.sort_values(0)

    def load_plm_vectors(self, retrieval_name_dir):
        aui_vectors = []

        assert os.path.exists(retrieval_name_dir), print(
            'No Vectors Saved. Check for naming errors.')

        print('Loading PLM Vectors.')
        files = glob(retrieval_name_dir + '/*')

        for i in range(len(files)):
            i_files = glob(retrieval_name_dir + '/*_{}.p'.format(i))
            if len(i_files) != 1:
                break
            else:
                aui_vectors.append(pickle.load(open(i_files[0], 'rb')))

        aui_vectors = np.vstack(aui_vectors)

        return aui_vectors

    def populate_vector_dictionary(self, aui_vectors):

        self.aui_vector_dict = {}
        for ind, aui in enumerate(self.sorted_umls_df.auis):
            self.aui_vector_dict[aui] = aui_vectors[ind]

    def compute_plm_vectors(self, retrieval_name_dir):
        self.plm.to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(self.retriever_name)

        all_cls = []

        with torch.no_grad():

            num_strings_proc = 0
            vec_save_batch_num = 0
            batch_sizes = []

            text_batch = []
            pad_size = 0

            curr_vecs = 0

            for i, row in tqdm(self.sorted_umls_df.iterrows(), total=len(self.sorted_umls_df)):

                string = str(row['strings'])
                length = row[0]

                text_batch.append(string)
                num_strings_proc += 1

                if length > pad_size:
                    pad_size = length

                if pad_size * len(text_batch) > 6000 or num_strings_proc == len(self.sorted_umls_df):

                    if not (os.path.exists(retrieval_name_dir + '/part_{}.p'.format(vec_save_batch_num))):
                        text_batch = list(text_batch)
                        encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True,
                                             max_length=self.plm.config.max_length)
                        input_ids = encoding['input_ids']
                        attention_mask = encoding['attention_mask']

                        input_ids = input_ids.to('cuda')
                        attention_mask = attention_mask.to('cuda')

                        outputs = self.plm(input_ids, attention_mask=attention_mask)
                        all_cls.append(outputs[0][:, 0, :].cpu().numpy())

                    batch_sizes.append(len(text_batch))
                    curr_vecs += 1

                    text_batch = []
                    pad_size = 0

                    if curr_vecs == 100:
                        print('Latest_batch_size {}'.format(batch_sizes[-1]))
                        print(sum(batch_sizes))
                        if not (os.path.exists(retrieval_name_dir + '/part_{}.p'.format(vec_save_batch_num))):
                            all_cls = np.vstack(all_cls)
                            pickle.dump(all_cls, open(retrieval_name_dir + '/part_{}.p'.format(vec_save_batch_num),
                                                      'wb'))

                        vec_save_batch_num += 1
                        all_cls = []
                        curr_vecs = 0

            if not (os.path.exists(retrieval_name_dir + '/part_{}.p'.format(vec_save_batch_num))):
                all_cls = np.vstack(all_cls)
                pickle.dump(all_cls, open(retrieval_name_dir + '/part_{}.p'.format(vec_save_batch_num),
                                          'wb'))

    def retrieve_knn(self):

        # Get Vectors
        original_auis = list(self.retriever_pipeline.original_auis)
        new_auis = list(self.retriever_pipeline.new_auis)

        original_vecs = []
        new_vecs = []

        for aui in original_auis:
            original_vecs.append(self.aui_vector_dict[aui])

        for aui in new_auis:
            new_vecs.append(self.aui_vector_dict[aui])

        original_vecs = np.vstack(original_vecs)
        new_vecs = np.vstack(new_vecs)

        # Preparing Data for k-NN Algorithm
        print('Chunking')

        dim = len(original_vecs[0])
        index_split = 3
        index_chunks = np.array_split(original_vecs, index_split)
        query_chunks = np.array_split(new_vecs, 100)

        k = self.retriever_pipeline.maximum_candidates_per_retriever

        # Building and Querying FAISS Index by parts to keep memory usage manageable.
        print('Building Index')

        index_chunk_D = []
        index_chunk_I = []

        current_zero_index = 0

        for num, index_chunk in enumerate(index_chunks):

            print('Running Index Part {}'.format(num))

            index = faiss.IndexFlatL2(dim)  # build the index

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

            D, I = [], []

            for q in tqdm(query_chunks):
                d, i = gpu_index.search(q, k)

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

        for D, I in zip(index_chunk_D, index_chunk_I):
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

        sorted_candidate_dictionary = {}

        for new_aui_index, nn_inds in tqdm(enumerate(full_sort_I)):
            nn_auis = [original_auis[i] for i in nn_inds]

            sorted_candidate_dictionary[new_auis[new_aui_index]] = nn_auis

        return sorted_candidate_dictionary

    def retrieve_scui_lui(self):
        """
        Use the SCUI (Source Synonymy) and LUI (Normalized Lexical Equivalence) chain to find potential synonyms.

        These two steps could be used alternatively for indefinitely many times until no more AUIs are to be added.
        For efficiency's sake, we will expand SCUIs from a LUI step and LUIs from an SCUI step.
        """

        sorted_candidate_dictionary = {}

        for aui in tqdm(self.retriever_pipeline.new_auis):
            scui = self.ontology.aui2scui[aui]
            lui = self.ontology.aui2lui

            # Expanding from SCUIs
            all_syns = []

            # Only get terms with source synonymy (Some terms have no SCUI)
            if scui.split('|||')[0] != '':
                source_syns = list(set(self.ontology.original_only_scui2auis.get(scui, [])))

                # For each source synonym, get all its originally defined synonyms and add them to the candidate list
                for source_syn_aui in source_syns:
                    source_syn_lui = self.ontology.aui2lui.get(source_syn_aui, None)

                    if source_syn_lui is not None:
                        lui_auis = self.ontology.original_only_lui2auis.get(source_syn_lui, [])
                        all_syns.extend(lui_auis)

            # Expanding from LUIs
            lui_syns = list(set(self.ontology.original_only_lui2auis.get(lui, [])))

            # For each source synonym, get all its originally defined synonyms and add them to the candidate list
            for lui_syn_aui in lui_syns:
                lui_syn_scui = self.ontology.aui2scui.get(lui_syn_aui, None)

                if lui_syn_scui is not None:
                    source_syn_auis = self.ontology.original_only_scui2auis.get(lui_syn_scui)
                    all_syns.extend(source_syn_auis)

            sorted_candidate_dictionary[aui] = all_syns

        return sorted_candidate_dictionary

    def retrieve_scui_cui(self):
        """
        Use the SCUI (Source Synonymy) and Original Synonyms to find potential synonyms.
        """

        sorted_candidate_dictionary = {}

        for aui in tqdm(self.retriever_pipeline.new_auis):
            scui = self.ontology.aui2scui[aui]

            if scui.split('|||')[0] != '':
                # Only get terms with source synonymy (Some terms have no SCUI)
                source_syns = list(set(self.ontology.original_only_scui2auis.get(scui, [])))

                # For each source synonym, get all its originally defined synonyms and add them to the candidate list
                all_syns = []
                for source_syn_aui in source_syns:
                    source_syn_cui = self.ontology.aui2cui[source_syn_aui]
                    original_syns = self.ontology.original_only_cui2auis.get(source_syn_cui, [])
                    all_syns.extend(original_syns)
            else:
                all_syns = []

            sorted_candidate_dictionary[aui] = all_syns

        return sorted_candidate_dictionary

    def retrieve_scui(self):
        """
        Use the SCUI (Source Synonymy) to find potential synonyms.
        """

        sorted_candidate_dictionary = {}

        for aui in tqdm(self.retriever_pipeline.new_auis):
            scui = self.ontology.aui2scui[aui]

            if scui.split('|||')[0] != '':
                # Only get terms with source synonymy (Some terms have no SCUI)
                source_syns = list(set(self.ontology.original_only_scui2auis.get(scui,[])))
            else:
                source_syns = []

            sorted_candidate_dictionary[aui] = source_syns

        return sorted_candidate_dictionary
