from absl import app, flags,logging
import os
import sys
#sys.path.append(os.path.dirname(os.getcwd()))
from gen_model_rep import gen_model_rep
from predict_pairs import predict_pairs
import os
import csv
import pickle
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.ops.gen_array_ops import unique, unique_eager_fallback
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Activation, Layer
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
import sys
import numpy as np
from math import ceil
import pandas as pd
import queue 
import time
#sys.path.append('/data/Bodenreider_UMLS_DL/ISWC2021')
#sys
#from .data.Bodenredifer_UMLS_DL.MedInfo2021 import create_model
#sys.path.append('/data/Bodenreider_UMLS_DL/MedInfo2021')
import run_umls_classifier_2 as ruc
from euc_common import NodeParallel, Utils
#from euc_common import Utils
import euc_run_data_generator as rdg
#from run_umls_classifier_iswc import fenizer, dump_pickle, load_pickle
import random
import tensorflow as tf
from collections import defaultdict 
from pathlib import Path
#FLAGS = flags.FLAGS
from gen_intersect_unique import gen_intersect_unique, gen_mrc_atom
FLAGS = flags.FLAGS


# flags.DEFINE_bool('v', True, 'some random test')
flags.DEFINE_string('umls_fp', "/data/Bodenreider_UMLS_DL/UMLS_VERSIONS/","file path to where all ULMS versions are stored")
flags.DEFINE_string('aa_umls_fn', "2020AA-ACTIVE" , "File name of older version of ELMS")
flags.DEFINE_string("ab_umls_fn", "2020AB-ACTIVE", "File name of newer version of ULMS")
flags.DEFINE_list('mrconso_fields', ['CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI', 'SCUI', 'SDUI','SAB','TTY', 'CODE', 'STR', 'SRL', "SUPPRESS", 'CVF'], 'keys for MRCONSO.RRF')

#filename for MRC_ATOMS pickle file
flags.DEFINE_string('aa_mrc_atoms_fn', 'AA_MRC_ATOMS.PICKLE', 'filename for MRC_ATOMS pickle for AA UMLS version')
flags.DEFINE_string('ab_mrc_atoms_fn', 'AB_MRC_ATOMS.PICKLE', 'filename for MRC_ATOMS pickle for AB UMLS version')

flags.DEFINE_string('unique_id2aui_pickle_fn', 'UNIQUE_ID2AUI.PICKLE', 'where to output new AUIs')
flags.DEFINE_string('intersect_id2aui_pickle_fn', 'INTERSECT_ID2AUI.PICKLE', 'where to output AUIs present in both ULMS versions')
flags.DEFINE_string('intersect_aui2id_pickle_fn', 'INTERSECT_AUI2ID.PICKLE', 'filename that stores intersect aui2id')
flags.DEFINE_string('unique_aui2id_pickle_fn', 'UNIQUE_AUI2ID.PICKLE', 'filename that stores unique aui2id')
flags.DEFINE_string('intersect_sg2id_pickle_fn', "INTERSECT_SG2ID.PICKLE", "filename for that stores list of aui id's in intersect that are of the same semantic group" )
flags.DEFINE_string('unique_sg2id_pickle_fn', "UNIQUE_SG2ID.PICKLE", "filename for that stores list of aui id's in unique that are of the same semantic group" )

flags.DEFINE_string('output_fp',os.environ.get('PWD') +'/', 'file path where pickle files will go')

flags.DEFINE_string('word_emb_variant', 'BioWordVec', 'BioBERT, BlueBERT, BERT, BioBERT_ULMS, UMLS')

flags.DEFINE_string('workspace_dir','/data/Bodenreider_UMLS_DL/', 'dir to the workspace')

#flags to run every stage
flags.DEFINE_boolean('gen_intersect_unique', False, 'finds intersection and unique AUIs')
flags.DEFINE_boolean('gen_model_rep', False, 'generates aui2vec of interseciton and unique AUIs')
flags.DEFINE_bool('validate_aui2layer', True, 'flag that runs test to determine if the auis in aui2layer are in correct order')
flags.DEFINE_boolean('evaluate', False, 'evaluates if split model aligns with output to original model')
flags.DEFINE_bool('predict', False, 'Flag to get predictions')
flags.DEFINE_bool('parallelize', False, 'parallelize predictions acorss nodes')

#defines which model you are evaluating
flags.DEFINE_string('model_type', 'medinfo', 'defines run_umls_classifer to import')

#path to model and model weights
flags.DEFINE_string('checkpoint_dp', '/data/Bodenreider_UMLS_DL/AAAI2022/UMLS_VERSIONS/2020AA-ACTIVE/NEGPOS1_WITHOUT_RAN/TRAINING/ALL_aaai2022_biowordvec_run2_lstm_attention_8192b_100ep_nodropout_exp1_modelTransE_SGD_variantAll_Triples_wordembBioWordVec/CHECKPOINT', 'file path to model weights to load')
flags.DEFINE_string('weights_fn', 'weights.100.hdf5', 'filename of desired weights to load')

#filename to save aui2layer pickle
flags.DEFINE_string('aui2layer_fn', 'AUI2LAYER.PICKLE', 'filename for unique AUI2layer')

flags.DEFINE_string('ab_aui2vec_fn', 'AB_AUI2VEC.PICKLE','pickle file to write 2020AB aui2vec')

flags.DEFINE_string('aui_info_fn', 'AUI_INFO.PICKLE', 'filename for aui_info, holds MRCONSO_MASTER fields')

flags.DEFINE_string("lstm_attention", "lstm_attention", "Attention, LSTM, or both")

flags.DEFINE_bool('use_sg', False, "form pairs from same semantic group" )

def euc_test_dict(d, dn=None):
    for i, (k,v) in enumerate(d.items()):
        if i < 2:
            logger.info('{} -> {}'.format(k,v))
    return


def euc_get_auis_str(atom_pickle_fp):
    """given mrc_atoms dictionary, returns seperate dictionary where key = (int)id value = string 

    Args:
        atom_pickle_fp (string): mrc_atoms filepath

    Returns:
        dict:  key = (int) id, value = string belonging to that AUI
    """
    auis_str = dict()
    aui_dict = ruc.load_pickle(atom_pickle_fp)

    for idx,v in enumerate(aui_dict.values()):
        auis_str[int(v['ID'])] = v['STR']
    euc_test_dict(auis_str)
    return auis_str

    

def euc_gen_tokenizer(tokenizer_fp,umls_new_fp, unique_pickle_fp):
    #if tokenizer doesn't exists create one with the updated ulms
    tokenizer = Tokenizer()
    tokenizer_output_fp = os.path.join(FLAGS.output_fp, FLAGS.tokenizer_pickle_fn)

    if(os.path.isfile(tokenizer_fp) is False):
        logging.info('creating new tokenizer...')
        logging.info('tokenizer_fp:{}'.format(tokenizer_fp))
        tokenizer = ruc.gen_tokenizer(umls_new_fp, tokenizer_output_fp)
    elif FLAGS.fit_tokenizer:
        #fit on unique AUIs
        logging.info('refitting tokenizer to new AUIs...')
        auis_str = euc_get_auis_str(unique_pickle_fp) #return a new dict using unique terms
        tokenizer = ruc.load_pickle(tokenizer_fp)
        tokenizer.fit_on_texts(auis_str.values())
        ruc.dump_pickle(tokenizer, tokenizer_output_fp)
    else:
        logging.info('fetching tokenizer...')
        tokenizer = ruc.load_pickle(tokenizer_fp)
    return tokenizer


def gen_partition(auis):
    """creates dictionary where key = tuples of idential aui_id pairs to extract n-1 layer output, value = arbitrary label

    Args:
        auis (list): list of auis 

    Returns:
        dict: key = tuple of auis, value = label 
    """
    partition = dict()
    for aui_id in auis:
        partition[(aui_id,aui_id)] = np.random.randint(0,2)
    logging.info('testing parition...')
    euc_test_dict(partition)
    return partition

def gen_val_partition(ab_aui2vec):
    """Generates dictionary where key = every possible pair that can be made from first aui in auis[], this serves to validate that we correctly extrated the 
    correct n-1 layer representation for each aui string

    Args:
        ab_aui2vec (dict): key = aui, value = embedding

    Returns:
        dict: key = tuple of auis, value = label
    """
    auis = list(ab_aui2vec.keys())
    partition = dict()
    aui_id_1 = auis[0]
    for i in range(1,len(auis)):
        partition[(aui_id_1,auis[i])] = np.random.randint(0,2)
        #partition[(aui_id_1, 8713194+436466+1+i)] = np.random.randint(0,2)
    return partition


def set_paths():
    paths = dict()
    paths['aa_umls_fp'] = os.path.join(FLAGS.umls_fp, FLAGS.aa_umls_fn) + "/META_DL/MRCONSO_MASTER.RRF"
    paths['ab_umls_fp'] = os.path.join(FLAGS.umls_fp, FLAGS.ab_umls_fn) + "/META_DL/MRCONSO_MASTER.RRF"

    #Atom pickle dictionaries
    #atom_path_old_fp = os.path.join(FLAGS.umls_fp, FLAGS.umls_fn_old) + "/META_DL/MRC_ATOMS.PICKLE"
    paths['aa_mrc_atoms_fp'] = os.path.join(FLAGS.output_fp, FLAGS.aa_mrc_atoms_fn) 
    paths['ab_mrc_atoms_fp'] = os.path.join(FLAGS.output_fp, FLAGS.ab_mrc_atoms_fn)
    
    paths['unique_id2aui_pickle_fp'] = os.path.join(FLAGS.output_fp,FLAGS.unique_id2aui_pickle_fn)
    paths['unique_aui2id_pickle_fp'] = os.path.join(FLAGS.output_fp, FLAGS.unique_aui2id_pickle_fn)
    paths['unique_sg2id_pickle_fp'] = os.path.join(FLAGS.output_fp, FLAGS.unique_sg2id_pickle_fn)

    paths['intersect_id2aui_pickle_fp'] = os.path.join(FLAGS.output_fp,FLAGS.intersect_id2aui_pickle_fn)
    paths['intersect_aui2id_pickle_fp'] = os.path.join(FLAGS.output_fp, FLAGS.intersect_aui2id_pickle_fn)
    paths['intersect_sg2id_pickle_fp'] = os.path.join(FLAGS.output_fp, FLAGS.intersect_sg2id_pickle_fn)


    #file path to existing tokenizer
    paths['tokenizer_pickle_fp'] = os.path.join(FLAGS.output_fp, FLAGS.tokenizer_pickle_fn) #taken from /data/Bodenreider_UMLS_DL/UMLS_VERSIONS/2020AA-ACTIVE/META_DL/
    
    #path to existing AUI2VEC file
    paths['ab_aui2vec_fp'] = os.path.join(FLAGS.output_fp, FLAGS.ab_aui2vec_fn)

    #path to word embeddings
    paths['embedding_fp'] = FLAGS.embedding_fp if FLAGS.embedding_fp is not None else os.path.join(FLAGS.workspace_dir, FLAGS.extra_dn, FLAGS.pre_trained_word2vec)

    #path to latest model weights
    paths['checkpoint'] = os.path.join(FLAGS.checkpoint_dp, FLAGS.weights_fn)
    paths['aui2layer_fp'] = os.path.join(FLAGS.output_fp, FLAGS.aui2layer_fn)
    
    paths['log_dp'] = os.path.join(FLAGS.output_fp, FLAGS.log_dn)
    Path(paths['log_dp']).mkdir(parents=True, exist_ok=True)
    paths['log_filepath'] = os.path.join(paths['log_dp'],"%s.log"%(FLAGS.application_name))
    
    paths['output_fp'] = FLAGS.output_fp
    paths['aui_info_fp'] = os.path.join(FLAGS.output_fp, FLAGS.aui_info_fn)
    return paths


def main(argv):

  #modifications to imported file functions
  global logger
  logger = logging
  #import_ruc()
  ruc.logger = logging
  ruc.test_dict = euc_test_dict
  #ruc.get_auis_str = euc_get_auis_str
  global utils
  utils = Utils(logger)
  paths = set_paths()

# file paths for different stages of script

  #full paths to MRCONSO_MASTER.RRF files


#stage 1) generate MRC_ATOM files if they don't exist and find unique and common AUIs
  if(FLAGS.gen_intersect_unique):   
      gen_intersect_unique(paths, logging)


#stage 2) generate aui2vec and get model representation of each aui in testset
  if(FLAGS.gen_model_rep):
    # model = tf.keras.Model()

    # tokenizer = ruc.load_pickle(paths['tokenizer_pickle_fp'])
    # embedding_matrix = ruc.make_word_embeddings(embedding_filepath=paths['embedding_fp'], vocab_length=len(tokenizer.word_index) + 1, tokenizer=tokenizer)
    # logging.info('embedding matrix size: {}'.format(np.shape(embedding_matrix)))
    # model = ruc.create_model(embedding_matrix)
    # model.load_weights(paths['checkpoint'])
    # logging.info('length of tokenizer before: {}'.format(len(tokenizer.word_index) + 1))
    # #tokenizer = gen_tokenizer(mod_tokenizer_fp, ulms_new_fp, unique_pickle_fp)
    # #logging.info('length of tokenizer after: {}'.format(len(tokenizer.word_index) + 1))
    # ab_aui2vec = euc_gen_aui2vec(tokenizer, paths['ab_aui2vec_fp'], paths['ab_ulms_fp']) 
    # logging.info('length of aui2vec: {}'.format(len(ab_aui2vec)))
    # model_mod = tf.keras.Model(inputs = model.inputs, outputs = [model.layers[-1]._inbound_nodes[0].input_tensors])
    # logging.info('getting layer output...')
    # partition = gen_partition(ab_aui2vec.keys())
    # generator = ruc.DataGenerator(partition, ab_aui2vec, None, FLAGS.max_seq_length, 
    #                                            FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
    #                                            batch_size = FLAGS.batch_size, 
    #                                            shuffle = False, is_test = False)

    # layer_output = get_model_output(generator, partition.values(), len(partition.values()), model_mod, FLAGS.batch_size, epoch = None, log_scores = None)
    # #TODO: compare left and right for all auis 
    # layer_output_left, layer_output_right = layer_output[0][0], layer_output[0][1]
    # logger.info('layer_output_left equal to {}'.format(np.array_equal(layer_output_left, layer_output_right)))
    # #logging.info('writing aui2layer...')
    #aui2layer = gen_aui2layer(layer_output_left, paths['aui2layer_fp'], ab_aui2vec)
   
    # TODO: generate unique aui2convec --> aui2convec = ruc.load_pickle(aui2convec_pickle_fp)

    #validate aui2layer
    # if(FLAGS.validate_aui2layer):
    #     #use ab_aui2vec and aui2layer and feed into model

    #     left_base_input = Input(shape = (50,), dtype = 'float32') 
    #     right_base_input = Input(shape = (50,), dtype = 'float32')
    #     mandist = ruc.ManDist()([left_base_input, right_base_input])
    #     model_manhat = tf.keras.Model(inputs=[left_base_input, right_base_input], outputs=mandist)       

    #     #mini testset
    #     partition = gen_val_partition(ab_aui2vec)


    #     generator = ruc.DataGenerator(partition, ab_aui2vec, None, FLAGS.max_seq_length, 
    #                                             FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
    #                                             batch_size = FLAGS.batch_size, 
    #                                             shuffle = False, is_test = True)
                                                
    #     model_pred = ruc.predict_generator(generator, partition.values(), len(partition.values()), FLAGS.batch_size, model, epoch = None, log_scores = None)

    #     generator = ruc.DataGenerator(partition, aui2layer, None, 50, 
    #                                             FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
    #                                             batch_size = FLAGS.batch_size, 
    #                                             shuffle = False, is_test = True)
        
        
    #     model_manhat_pred = ruc.predict_generator(generator, partition.values(), len(partition.values()), FLAGS.batch_size, model_manhat, epoch = None, log_scores = None)

    #     logging.info('shape of model_pred: {}'.format(np.shape(model_pred)))
    #     logging.info('shape of model_manhat_pred = {}'.format(np.shape(model_manhat_pred)))

    #     if np.array_equal(model_pred, model_manhat_pred):
    #         logger.info('Validated AUI2LAYER')
    #     else:
    #         logger.info('AUI2LAYER not producing correct scores, revisit implementation')
    # #Stage 3: Generate pairs and predict 
    gen_model_rep(paths, logger)
  if FLAGS.predict == True:
      predict_pairs(paths, logger)
    #paths = dict()

    #For storing the logs 
    # paths['log_dp'] = os.path.join(FLAGS.output_fp, FLAGS.log_dn)
    # Path(paths['log_dp']).mkdir(parents=True, exist_ok=True)
    # paths['log_filepath'] = os.path.join(paths['log_dp'],"%s.log"%(FLAGS.application_name))
    # utils = Utils()
    # logger = utils.get_logger(logging.INFO, FLAGS.application_name, paths['log_filepath'])
    # utils.set_logger(logger)
    # rdg.set_utils(utils)

    # #For the executable files
    #paths['output_fp'] = FLAGS.output_fp
    # Path(paths['bin_dp']).mkdir(parents=True, exist_ok=True)

    #generating neccessary files 
    # aui_info = rdg.get_aui_info_gen_neg_pairs( os.path.join(FLAGS.output_fp, FLAGS.ab_mrc_atoms_fn) , os.path.join(FLAGS.output_fp, FLAGS.aui_info_fn), utils)
    
    # #cui2aui_id = rdg.get_cui2aui_id_dict(os.path.join(FLAGS.umls_fp, FLAGS.ab_umls_fn + "/META_DL/MRCONSO_MASTER.RRF"), os.path.join(FLAGS.output_fp, FLAGS.cui2aui_id_fn),utils)
  
    # cui2aui_id = dict()
    

    
    # left_base_input = Input(shape = (50,), dtype = 'float32') 
    # right_base_input = Input(shape = (50,), dtype = 'float32')
    # mandist = ruc.ManDist()([left_base_input, right_base_input])
    # model_manhat = tf.keras.Model(inputs=[left_base_input, right_base_input], outputs=mandist) 


    # if FLAGS.parallelize == True:
    #      rdg.parallel_predict_pairs(FLAGS.job_name, paths,logger)
        
    # else:
    #     rdg.gen_pairs_predict_batch(FLAGS.start_idx, FLAGS.end_idx, aui_info, cui2aui_id, logger, FLAGS.n_processes)
   

        



    #create model
    # embedding_matrix = None
    # if FLAGS.word_emb_variant == "BioWordVec":    #default   
    #     embedding_matrix = ruc.make_word_embeddings(embedding_fp, len(tokenizer.word_index) + 1, tokenizer)

    # elif FLAGS.word_emb_variant == "BERT":
    #     embedding_matrix = ruc.make_bert_embedding_matrix(embedding_fp)  



    

#stage 3)  #Make predictions
    #TODO: Look into predict_generator in run_umls_classifier.py

  






if __name__ == '__main__':
  app.run(main)
