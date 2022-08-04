import _pickle as pickle
import pandas as pd
from tqdm import tqdm
import time
from RetrievalPipeline import RetrievalPipeline
import numpy as np
from SynonymyClassifier import SynonymyClassifier, SynonymyDatasetManager

class UMLS:
    """
    Class designed to hold and manage the UMLS Ontology.
    """

    def __init__(self, umls_directory='/data/Bodenreider_UMLS_DL/UMLS_VERSIONS', version='2020AB'):

        self.version = '2020AB'
        self.directory = umls_directory

        # Load all UMLS Info Necessary
        self.aui_info = []


        # CUI-AUI Mapping
        self.cui2auis = {}
        self.aui2cui = {}
        self.cui_aui = []

        # AUI-Source CUI Mapping
        self.aui2scui = {}
        self.scui2auis = {}

        # Preferred Name Mapping
        self.cui2preferred = {}

        # AUI-String Mapping
        self.aui2str = {}
        self.str2aui = {}

        # Semantic Type & Group Mapping
        self.aui2sg = {}
        self.cui_sg = []
        self.cui2sg = {}
        self.semtype2sg = {}

        # AUI-LUI Mapping
        self.aui2lui = {}
        self.lui2auis = {}

        self.original_only_lui2auis = None
        self.original_only_scui2auis = None
        self.original_only_cui2auis = None

        # Loading UMLS Info from File
        self.raw_load_umls(umls_directory, version)
        self.get_relevant_aui_set()
        self.create_mappings()

    def raw_load_umls(self,
                      umls_directory,
                      version):

        print('Loading Raw MRCONSO Lines')
        # Download all MRCONSO
        with open('{}/{}-ACTIVE/META/MRCONSO.RRF'.format(umls_directory, version), 'r') as fp:
            pbar = tqdm(total=15000000)
            line = fp.readline()

            while line:
                line = line.split('|')
                cui = line[0]
                language = line[1]
                term_status = line[2]
                lui = line[3]
                string_type = line[4]

                aui = line[7]
                scui = line[9]
                source = line[11]
                string = line[-5]

                self.aui_info.append((aui, cui, string, scui, source, term_status, string_type, language, lui))

                line = fp.readline()
                pbar.update(1)

        # Make sure SemGroups.txt file exists
        self.sem_groups = pd.read_csv('SemGroups.txt', sep='|', header=None)

        # Map Semantic Types to Semantic Groups
        for i, row in self.sem_groups.iterrows():
            st = row[3]
            sg = row[1]

            self.semtype2sg[st] = sg

        # Download all MRSTY and populate dictionary
        with open('{}/{}-ACTIVE/META/MRSTY.RRF'.format(umls_directory, version), 'r') as fp:

            for line in fp.readlines():
                line = line.split('|')
                cui = line[0]
                st = line[3]
                sg = self.semtype2sg[st]

                self.cui2sg[cui] = sg

    def get_relevant_aui_set(self,
                             original_auis_filename='/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/INTERSECT_AUI2ID.PICKLE',
                             new_auis_filename='/data/Bodenreider_UMLS_DL/Interns/Vishesh/eval_umls/UNIQUE_AUI2ID.PICKLE'):

        # Load Original and New AUIs (Synonyms obtained only from AUIs in these 2 sets)
        original_auis = pickle.load(open(original_auis_filename, 'rb'))
        new_auis = pickle.load(open(new_auis_filename, 'rb'))

        self.relevant_auis = set(original_auis).union(new_auis)

        return original_auis, new_auis

    def create_mappings(self):
        print('Creating mappings between concept IDs for easy access.')

        for tup in tqdm(self.aui_info):
            current_time = time.time()

            aui = tup[0]
            cui = tup[1]
            string = tup[2]
            scui = tup[3] + '|||' + tup[4]  # Source CUI Uniqueness is confined to each source
            pref = (tup[5], tup[6])
            lang = tup[7]
            lui = tup[8]

            sg = self.cui2sg[cui]

            self.aui2scui[aui] = scui

            if pref[0] == 'P' and pref[1] == 'PF' and lang == 'ENG':
                self.cui2preferred[cui] = string

            self.aui2str[aui] = string
            self.aui2cui[aui] = cui
            self.aui2sg[aui] = sg
            self.aui2lui[aui] = lui

            auis = self.str2aui.get(string, [])
            auis.append(aui)
            self.str2aui[string] = auis

            self.cui_sg.append((cui, sg))
            self.cui_aui.append((cui, aui))

            # Only Obtain Synonyms from AUIs defined
            if aui in self.relevant_auis:
                auis = self.cui2auis.get(cui, [])
                auis.append(aui)
                self.cui2auis[cui] = auis

                auis = self.scui2auis.get(scui, [])
                auis.append(aui)
                self.scui2auis[scui] = auis

                auis = self.lui2auis.get(lui, [])
                auis.append(aui)
                self.lui2auis[lui] = auis

                if (time.time() - current_time) > 5:
                    print(tup)

    def get_original_ontology_synonyms(self, original_auis):
        """
        Build CUI to AUI set, SCUI to AUI and LUI to AUI set mappings
        which only contain AUIs from the "original" ontology.
        """

        self.original_only_cui2auis = {}
        self.original_only_scui2auis = {}
        self.original_only_lui2auis = {}

        for aui in tqdm(original_auis):
            cui = self.aui2cui[aui]
            scui = self.aui2scui[aui]
            lui = self.aui2lui[aui]

            auis = self.original_only_cui2auis.get(cui, [])
            auis.append(aui)
            self.original_only_cui2auis[cui] = auis

            auis = self.original_only_scui2auis.get(scui, [])
            auis.append(aui)
            self.original_only_scui2auis[scui] = auis

            auis = self.original_only_lui2auis.get(lui, [])
            auis.append(aui)
            self.original_only_lui2auis[lui] = auis

    def augment_umls(self,
                     original_aui_filename,
                     new_aui_filename,
                     output_dir,
                     retriever_names,
                     maximum_candidates,
                     classifier_name,
                     candidates_to_classify=100,
                     add_gold_candidates=False,
                     dev_perc=0.0,
                     test_perc=0.0):
        """
        This method enables new terms to be introduced automatically into the UMLS Ontology using only their strings
        and any synonymy information available from its source ontology.

        Args:
            original_aui_filename: Filename with a list of AUIs to use as original or "base" ontology.
            (Pickle file w/ iterable)
            new_aui_filename: Filename with a list of AUIs that are to be introduced into the original or "base"
            ontology. (Pickle file w/ iterable)
            retriever_names: List of names for all retriever systems to be used in producing candidate AUIs from the
            original ontology for each new AUIs.
            maximum_candidates: Maximum # candidates per retrieval method
            classifier_name: Name of classification system (For now the location of a fine-tuned PLM model)
            candidates_to_classify: # of candidates to pass through classification system
            add_gold_candidates: Whether to add synonyms to retrieved candidates (Defaults to no for inference run)
            dev_perc: Percentage of total concepts to introduce into the development set
            test_perc: Percentage of total concepts to introduce into the test set

        Returns:
            predicted_synonymy: Dictionary linking each new AUI to a list of AUIs from the original ontology
            which are predicted to be synonymous.
            evalution_metrics:
                - Recall @ Various N's for each retriever
                - F1, Precision and Recall for final classifier
        """

        # Create Retrieval Pipeline
        self.retriever_pipeline = RetrievalPipeline(original_aui_filename,
                                                    new_aui_filename,
                                                    self,
                                                    output_dir,
                                                    maximum_candidates,
                                                    retriever_names)

        # Run Candidate Generation (Retrieval)
        self.retriever_pipeline.run_retrievers()

        # Create Classification Dataset Using Candidates Extracted (kNN on new AUIs)
        self.dataset_manager = SynonymyDatasetManager(self,
                                                      self.retriever_pipeline,
                                                      self.retriever_pipeline.output_dir,
                                                      candidates_to_classify,
                                                      dev_perc,
                                                      test_perc,
                                                      add_gold_candidates)
        self.dataset_manager.create_classification_dataset()

        # self.classifier = SynonymyClassifier(self, classifier_name,candidates_to_classify)

        # Run on Test Set

    def verify_umls(self,
                    retriever_names,
                    maximum_candidates,
                    re_ranker_name):
        """
        This method enables the automatic verification of already existing UMLS synonymy. Same as the "augment_umls"
        method except that the original and new ontologies are equivalent, meaning that each

        Args:
            original_aui_filename: Filename with a list of AUIs to use as original or "base" ontology.
            new_aui_filename: Filename with a list of AUIs that are to be introduced into the original or "base"
            ontology.
            retriever_names: List of names for all retriever systems to be used in producing candidate AUIs from the
            original ontology for each new AUIs.
            maximum_candidates: Maximum # candidates per retrieval method
            re_ranker_name: Name of re-ranking system (For now a fine-tuned PLM model)

        Returns:
            predicted_synonymy: Dictionary linking each new AUI to a list of AUIs from the original ontology
            which are predicted to be synonymous.

        """
        pass
