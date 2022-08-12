import csv
import re
import sys
import os
import math
import random
import shutil
import gc
import time
from pathlib import Path
import glob
from multiprocessing import Process, Queue, Manager
import queue
import copy

import numpy as np
from tqdm import tqdm
import bisect
import subprocess
import logging
import absl

from absl import app
from absl import flags
import run_umls_classifier_2 as ruc
from euc_common import SlurmJob, Utils, NodeParallel
import tensorflow as tf
from tensorflow.keras.layers import Input

FLAGS = flags.FLAGS
#flags.DEFINE_string('f', '', 'kernel')

flags.DEFINE_string('server','FM','FM, Biowulf')
flags.DEFINE_string('application_name', 'euc_run_data_generator', '')
flags.DEFINE_string('application_py_fn', 'test_2.py', '')
flags.DEFINE_string('scores_fn', 'scores.csv', 'filename to write scores to')

#flags.DEFINE_string('workspace_dp','/nfsvol2/projects/umls/deeplearning/aaai2020/2019AB','')
#flags.DEFINE_string('important_info_fn','IMPORTANT_INFO.RRF','')
#flags.DEFINE_string('umls_version_dp','/nfsvol2/projects/umls/deeplearning/aaai2020/2019AB/UMLS_VERSIONS/2020AA-ACTIVE','')
#flags.DEFINE_string('dataset_version_dn','NEGPOS3', '')
# flags.DEFINE_string('umls_version_dp','/nfsvol2/projects/umls/deeplearning/aaai2020/2019AB/UMLS_VERSIONS/2020AA-ACTIVE','')
#flags.DEFINE_string('umls_dl_dp', None, '')
#flags.DEFINE_string('umls_dl_dn','META_DL', '')
#flags.DEFINE_string('umls_meta_dp', None, '')
#flags.DEFINE_string('umls_meta_dn','META','')
#flags.DEFINE_string('datasets_dp', None, '')
#flags.DEFINE_string('datasets_dn', "datasets", '')
flags.DEFINE_string('log_dn', 'logs','')
#flags.DEFINE_string('bin_dn', 'bin','')
#flags.DEFINE_string('extra_dn', 'extra', '')
# flags.DEFINE_string('workspace_dp','/data/nguyenvt2/aaai2020','')
# flags.DEFINE_string('umls_version_dp','/data/nguyenvt2/aaai2020/data/META_2019AB','')

flags.DEFINE_bool('gen_master_file', True, 'Your name.')
# flags.DEFINE_bool('gen_master_file', False, 'Your name.')
flags.DEFINE_bool('gen_pos_pairs', True, 'Your name.')
# flags.DEFINE_bool('gen_pos_pairs', False, 'Your name.')
flags.DEFINE_bool('gen_swarm_file', True, 'Your name.')
# flags.DEFINE_bool('gen_swarm_file', False, 'Your name.')
# flags.DEFINE_bool('exec_gen_neg_pairs_swarm', True, '')
flags.DEFINE_bool('exec_gen_neg_pairs_swarm', False, '')
# flags.DEFINE_bool('gen_neg_pairs', True, 'Your name.')
flags.DEFINE_bool('gen_neg_pairs', False, 'Your name.')
# flags.DEFINE_bool('gen_neg_pairs_batch', True, 'Your name.')
flags.DEFINE_bool('gen_neg_pairs_batch', False, 'Your name.')
flags.DEFINE_bool('gen_dataset', True, 'N')
# flags.DEFINE_bool('gen_dataset', False, 'N')
#flags.DEFINE_bool('run_slurm_job', True, '')
flags.DEFINE_bool('run_slurm_job', False, '')

flags.DEFINE_string('dataset_dp', None, '')
flags.DEFINE_string('dataset_dn', None, '')
flags.DEFINE_string('mrconso_fn','MRCONSO.RRF','')
flags.DEFINE_string('cui_sty_fn','MRSTY.RRF','')
flags.DEFINE_string('mrxns_fn','MRXNS_ENG.RRF','')
flags.DEFINE_string('mrxnw_fn','MRXNW_ENG.RRF','')
flags.DEFINE_string('sg_st_fn','SemGroups.txt','')

flags.DEFINE_string('mrx_nw_id_fn','NW_ID.RRF','')
flags.DEFINE_string('mrx_ns_id_fn','NS_ID.RRF','')
flags.DEFINE_string('nw_id_aui_fn','NW_ID_AUI.RRF','')
#flags.DEFINE_string('mrconso_master_fn','MRCONSO_MASTER.RRF','')
flags.DEFINE_string('mrconso_master_randomized_fn','MRCONSO_MASTER_RANDOMIZED.RRF','')
flags.DEFINE_string('aui_info_gen_neg_pairs_pickle_fn', 'AUI_INFO_GEN_NEG_PAIRS.PICKLE', '')
flags.DEFINE_string('cui_to_aui_id_pickle_fn', 'CUI_AUI_ID.PICKLE', '')
#flags.DEFINE_string('inputs_pickle_fn','INPUTS_DATA_GEN.PICKLE','')
flags.DEFINE_string('inputs_keys_ext', '_keys', '')
flags.DEFINE_string('inputs_pickle_fp', None, '')
#flags.DEFINE_list('mrconso_master_fields', ["ID", "CUI", "LUI", "SUI", "AUI", "AUI_NUM", "SCUI", "NS_ID", "NS_LEN", "NS", "NORM_STR","STR", "SG"], '')
#flags.DEFINE_list('mrconso_fields', ["CUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE","STR","SRL","SUPPRESS","CVF"], '')
#flags.DEFINE_list('ds_fields', ['jacc', 'AUI1', 'AUI2', 'Label'],'')
#flags.DEFINE_string("aui2id_pickle_fn", 'AUI2ID.PICKLE', "The output directory where the model checkpoints will be written.")
#flags.DEFINE_string("id2aui_pickle_fn", 'ID2AUI.PICKLE', "The output directory where the model checkpoints will be written.")

flags.DEFINE_string('pos_pairs_fn','POS.RRF','')
flags.DEFINE_string('neg_file_prefix','NEG_FILES','')
flags.DEFINE_string('neg_batch_file_prefix','NEG_BATCH_FILES','')
flags.DEFINE_string('completed_inputs_fn','COMPLETED_INPUTS.RRF','')
flags.DEFINE_string('completed_inputs_fp', None, '')

flags.DEFINE_integer('gen_fold', 1, '')
#flags.DEFINE_string('train_fn', 'TRAIN_DS.RRF', "The output directory where the model checkpoints will be written.")
#flags.DEFINE_string('dev_fn', 'DEV_DS.RRF', "The output directory where the model checkpoints will be written.")
#flags.DEFINE_string('test_fn', 'TEST_DS.RRF', "The output directory where the model checkpoints will be written.")

flags.DEFINE_string('swarm_fn','gen_neg_pairs.swarm','')
flags.DEFINE_string('submit_gen_neg_pairs_jobs_fn','gen_neg_pairs.submit','')
flags.DEFINE_string('data_generator_fn','run_data_generator.py','')

flags.DEFINE_bool('gen_neg_pairs_flavor_topn_sim', True, 'N')
flags.DEFINE_bool('gen_neg_pairs_flavor_ran_sim', True, '')
flags.DEFINE_bool('gen_neg_pairs_flavor_ran_nosim', True, '')

flags.DEFINE_string('training_type', "LEARNING_DS", '')
flags.DEFINE_string('gentest_type', "GENTEST_DS", '')

flags.DEFINE_string('neg_pairs_flavor_topn_sim', 'TOPN_SIM', 'TOPN_SIM, RAN_SIM, RAN_NOSIM')
flags.DEFINE_string('neg_pairs_flavor_ran_sim', 'RAN_SIM', '')
flags.DEFINE_string('neg_pairs_flavor_ran_nosim', 'RAN_NOSIM', '')
flags.DEFINE_string('neg_pairs_flavor_all', 'ALL', '')
flags.DEFINE_string('neg_pairs_flavor_ran', 'RAN', '')
# For SLURM
flags.DEFINE_string('user','nguyenvt2', '')
flags.DEFINE_string('job_name', 'data_gen', '')
flags.DEFINE_string('job_id', None, '')
flags.DEFINE_string('conda_env', 'tf_220_1', '')
flags.DEFINE_bool('debug', False, '')

flags.DEFINE_integer('ntasks', 0, '')
flags.DEFINE_integer('n_processes', 3, '') #TODO: Change back to 16 by default 
flags.DEFINE_integer('ram', 150, '')
flags.DEFINE_integer('time_limit', None, '')
flags.DEFINE_integer('interval_time_between_create_process_task', 10, '')
flags.DEFINE_integer('interval_time_check_gen_neg_pairs_complete_task', 10, '')
flags.DEFINE_integer('interval_time_check_gen_neg_pairs_task', 10, '')

flags.DEFINE_integer('neg_to_pos_rate', 1, 'Your name.')
flags.DEFINE_integer('start_idx', 0, '')
flags.DEFINE_integer('end_idx', 4, '')
#flags.DEFINE_string('delimiter', '|', '')
flags.DEFINE_integer('shuffle_cnt', 3, '')
flags.DEFINE_integer('ninputs', 0, 'number of auis in UNIQUEAUI2ID to generate pairs from')

flags.DEFINE_bool('raw', False, 'flag to fetch raw predictions for every AUI')

csv.field_size_limit(sys.maxsize)

def set_utils(l_utils):
    global utils 
    utils = l_utils

def get_aui_lst(input_aui, nw_id_aui_dict, aui_info):
    input_aui_ns_id = aui_info[input_aui]["NS_ID"]
    aui_lst = []
    for nw in input_aui_ns_id:
        aui_lst += nw_id_aui_dict[int(nw)]  
    return list(set(aui_lst))



def get_aui_info_gen_neg_pairs(mrc_atoms_fp, aui_info_pickle_fp,utils):

    aui_info = dict()
    
    if os.path.isfile(aui_info_pickle_fp):
        aui_info = utils.load_pickle(aui_info_pickle_fp)
    else:
        mrc_atoms = utils.load_pickle(mrc_atoms_fp)
        with tqdm(total = len(mrc_atoms)) as pbar:
            for aui, row in mrc_atoms.items():
                pbar.update(1)
                # id_aui_dict = dict()
                # id_aui_dict['CUI'] = row['CUI']
                # id_aui_dict['AUI'] = aui
                aui_info[row['ID']] = {"CUI" : row['CUI'], "AUI": aui, "SG": row["SG"]}
        utils.dump_pickle(aui_info, aui_info_pickle_fp)
        del mrc_atoms

    return aui_info

def update_score_raw(aui_id1, predictions):
    """creates histogram array where bins are (<0.4990, [0.4990, 0.5], >0.5). This
    is to evaluate if there exists auis whose raw values are at risk of being round to 0 and 1 inconsistently

    Args:
        aui_id1 (int): AUI ID with which pairs we generated
        predictions (double): continous prediction values [0,1.0]
    """
    bins = [0.0, 0.4990, 0.4995, 0.4999, 0.5000, 0.5001, 1.0]
    hist,edges = np.histogram(predictions, bins)
    edges = ['[' + str(edges[i]) + ',' + str(edges[i+1]) + ')' for i in range(len(edges)-1)]
    with open(FLAGS.scores_fn, mode='a') as scores_file:
        fieldnames = ['AUI_ID'] + edges
        writer = csv.DictWriter(scores_file, fieldnames=fieldnames)
        if scores_file.tell() == 0:
            writer.writeheader()
        row = {key:val for key,val in zip(edges, hist)}
        row.update({'AUI_ID': aui_id1})
        writer.writerow(row)

def update_score(aui_id_1, y_pred, y_true, pairs, sg):
    TP, TN, FP, FN = 0, 0, 0, 0
    #err = list()
    for i, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == 1 and p == 1:
            TP += 1
        elif t == 1 and p == 0:
            FN += 1
            #err.append(pairs[i])
        elif t == 0 and p == 1:
            FP += 1
            #err.append(pairs[i])
        elif t== 0 and p == 0:
            TN += 1
        else:
            print("unable to update score")
    
    #update file 
    #TODO: have path to csv be a funciton argument
    with open(FLAGS.scores_fn, mode='a') as scores_file:
        fieldnames = ['AUI_ID', 'TP', 'TN', 'FP', 'FN', "SG"]
        writer = csv.DictWriter(scores_file, fieldnames=fieldnames)
        if scores_file.tell() == 0:
            writer.writeheader()
        writer.writerow({'AUI_ID': aui_id_1, 'TP': TP, 'TN': TN, 'FP':FP,'FN':FN, "SG":sg})
    # print('length of err: {}'.format(len(err)))
    # print('FN: {}'.format(FN))
    # with open('errors.csv', 'a') as f:
    #      writer = csv.DictWriter(f, fieldnames=['aui1','aui2'])
    #      if f.tell() == 0:
    #         writer.writeheader()
    #      for aui1,aui2 in err:
    #         writer.writerow({'aui1':aui1, 'aui2':aui2})
    #         f.write("%s, %s\n" % (str(aui1), str(aui2)))
    del TP 
    del TN 
    del FP
    del FN
    return

def compute_pairs(input_aui_id, input_aui_idx, intersect_ids, unique_ids, aui_info):
    pairs = dict()

    for id2 in intersect_ids:
        if aui_info[input_aui_id]['CUI'] == aui_info[id2]['CUI']:
            pairs[(input_aui_id, id2)] = 1
        else:
            pairs[(input_aui_id, id2)] = 0
            
    #element wise product between input_aui_id and all other auis in AB-specific
    for aui_idx in range(input_aui_idx+1, len(unique_ids)):
        if aui_info[input_aui_id]['CUI'] == aui_info[unique_ids[aui_idx]]['CUI']:
            pairs[(input_aui_id, unique_ids[aui_idx])] = 1
        else:
            pairs[(input_aui_id, unique_ids[aui_idx])] = 0
    
        
        # if len(pairs) != len(intersect_ids)+ len(unique_ids)-input_aui_idx-1:
        #         print("mismatch")


    Utils.test_dict(pairs)
    return pairs

def generate_pairs_predict(input_queue, being_processed, completed, **kwargs):
    '''Generates all pairs, runs predictions, then writes to file '''

    intersect = kwargs['intersect']
    start_time = time.time()
    aui_info = kwargs['aui_info']
    unique = kwargs['unique']
    batch_size = kwargs['batch_size']
    aui2layer = kwargs['aui2layer']
    #model_manhat = kwargs['model_manhat']
    #ab_aui2vec = kwargs['ab_aui2vec']
    
    # intersect_id2aui = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.intersect_id2aui_pickle_fn))
    # aui_info =  ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.aui_info_fn))
    # cui_to_aui_dict = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.cui2aui_fn))
    # unique_id2aui = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.unique_id2aui_pickle_fn))
    # aui2layer = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.aui2layer_fn))
    # batch_size = 8192

    num = 0
    p_start = time.time()
    start_time = time.time()
    p_id = os.getpid()
    cnt_sim = len(intersect)
    #generator, pairs, predictions, idx, aui_id1 = 0,0,0,0,0
    while (input_queue.empty() is False):

        left_base_input = Input(shape = (50,), dtype = 'float32') 
        right_base_input = Input(shape = (50,), dtype = 'float32')
        mandist = ruc.ManDist()([left_base_input, right_base_input])
        model_manhat = tf.keras.Model(inputs=[left_base_input, right_base_input], outputs=mandist) 

        start_time = time.time()
        idx, aui_id1 = input_queue.get()
        # if aui1 == 'END':
        #     break
        #print('appending to being_processed dictionary')
        being_processed[(idx, aui_id1)] = p_id
        #n = int(aui_info[aui1]["AUI_NUM"])-1
        #logger.info("aui_info[{}]: {} with n: {}".format(aui1, aui_info[aui1], n))
        # if n == 0:
        #     n = 1 # To cover all AUIs
        # k = neg_to_pos_rate*n
        #cnt_sim = len(intersect_ids)
          
        #Generate pairs and labels for positive and negative pairs 
        pairs = list()
        if FLAGS.use_sg:
            sg = aui_info[aui_id1]["SG"]
            pairs = compute_pairs(aui_id1, idx, intersect[sg], unique[sg], aui_info)
        else:
            pairs = compute_pairs(aui_id1, idx, list(intersect.keys()), list(unique.keys()), aui_info)
        # print('computed pairs')
        # print('length of pairs = {}'.format(len(pairs)))
        
        # print('pairs: ')
        # print(pairs)
        
        # print('aui2layer: ')
        # print(aui2layer)
        
        generator = ruc.DataGenerator(pairs, aui2layer, None, 50, 
                                                FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
                                                batch_size = FLAGS.batch_size, 
                                                shuffle = False, is_test = True) #set to false to actually use our labels
        
        # #push to output queue for model to take and predict 
        
        # generator = ruc.DataGenerator(pairs, ab_aui2vec, None, FLAGS.max_seq_length, 
        #                                         FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
        #                                         batch_size = FLAGS.batch_size, 
        #                                         shuffle = False, is_test = True)
         
         
         #original medinfo model, for testing purposes, remove once done
        # embedding_fp = FLAGS.embedding_fp if FLAGS.embedding_fp is not None else os.path.join(FLAGS.workspace_dir, FLAGS.extra_dn, FLAGS.pre_trained_word2vec)
        # tokenizer = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.tokenizer_pickle_fn)) 
        # embedding_matrix = ruc.make_word_embeddings(embedding_filepath=embedding_fp, vocab_length=len(tokenizer.word_index) + 1, tokenizer=tokenizer)

        # logging.info('embedding matrix size: {}'.format(np.shape(embedding_matrix)))
        # model = tf.keras.Model()
        # model = ruc.create_model(embedding_matrix)
      
        predictions = ruc.predict_generator(generator, pairs.values(), len(pairs), FLAGS.batch_size, model_manhat, raw=FLAGS.raw)

        if FLAGS.raw:
            update_score_raw(aui_id1, predictions)
        else:
            update_score(aui_id1, predictions, list(pairs.values()), list(pairs.keys()), aui_info[aui_id1]["SG"])
        completed['queue'].put(str(aui_id1))
        del being_processed[(idx, aui_id1)]
        del model_manhat    
        del generator
        del pairs
        del idx
        del aui_id1
        del predictions
        gc.collect() #resolves memory leak in model.predict() 
        tf.keras.backend.clear_session()
        
        
    completed['status'] = True
    return

def parallel_predict_pairs(job_name, paths, logger):
    
    submit_parameters = [
        " -b 1",
        " --merge-output"
        " -g " + str(FLAGS.ram),
        " -t " + str(FLAGS.n_processes), #number of CPUs per subjob
        " --time 2-00:00:00 --logdir %s"%(os.path.join(FLAGS.output_fp, FLAGS.log_dn, FLAGS.job_name)),
    ]
    prepy_cmds = ['source /data/javangulav2/conda/etc/profile.d/conda.sh',
                'conda activate %s'%FLAGS.conda_env]
    paths['execute_py_fp'] = os.path.join(FLAGS.output_fp, FLAGS.application_py_fn)

    unique_id2aui = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.unique_id2aui_pickle_fn))

    swarm_parameters = [
        " --workspace_dp=%s"%FLAGS.workspace_dp,
        " --umls_version_dp=%s"%FLAGS.umls_version_dp,
        " --umls_dl_dp=%s"%FLAGS.umls_dl_dp,
        " --n_processes=%d"%FLAGS.n_processes,
        " --job_name=%s"%(job_name),
        " --predict=true",
        " --parallelize=false",
        " --start_idx=start_index",
        " --end_idx=end_index",
        " --dataset_version_dn=%s"%(FLAGS.dataset_version_dn),
        " --conda_env=%s"%FLAGS.conda_env,
        " --gen_intersect_unique=false",
        " --gen_model_rep=false",
        "--scores_fn=%s"%FLAGS.scores_fn,
        "--raw=%s"%FLAGS.raw,
        "--use_sg=%s"%FLAGS.use_sg
        # " --debug=%s"%debug,
        # " --inputs_pickle_fp=%s"%paths['inputs_pickle_fp'],
        # " --completed_inputs_fp=%s"%(paths['completed_inputs_fp']),
        # " > %s"%(os.path.join(paths['data_gen_log_dp'], "%s_start_index_end_index.log "%(job_name))),
        ]

    # input_paras = {'inputs': paths['inputs_pickle_fp'],
    #             'inputs_keys': paths['inputs_pickle_fp'] + FLAGS.inputs_keys_ext}

    # output_paras = {'training_type': paths['neg_pairs_training_flavor_fp'], 
    #                 'gentest_type': paths['neg_pairs_gentest_flavor_fp']}
    # output_globs = {'training_type': paths['neg_pairs_training_flavor_glob'], 
    #                 'gentest_type': paths['neg_pairs_gentest_flavor_glob']}

    '''for testing'''
    #ninputs = len(unique_id2aui)
    #logger.info('ninputs = {}'.format(ninputs))
    ninputs = len(unique_id2aui) if FLAGS.ninputs == 0 else FLAGS.ninputs
    ntasks = 1 if FLAGS.ntasks == 0 else FLAGS.ntasks #total number of subjobs

    

    slurmjob = SlurmJob(job_name, True, prepy_cmds, swarm_parameters, submit_parameters, ntasks, ninputs, None, paths, logger=logger)
    slurmjob.run()

def gen_pairs_predict_batch(start_idx, end_idx, aui_info, cui2aui_id, logger, n_processes=2):
    '''Takes in index range to create queue of auis from 2020AB-specific, which will form the left hand side of each pair'''

    start_time = time.time()


    #unique_id2aui = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.unique_id2aui_pickle_fn))
    intersect_id2aui = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.intersect_id2aui_pickle_fn))
    aui_info = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.aui_info_fn))
    
    if FLAGS.debug is True:
        logger.info("Loading input to queue...")
    input_queue = queue.Queue()
    unique_id2aui = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.unique_id2aui_pickle_fn))
    kwargs = {'intersect': intersect_id2aui, 'aui_info':aui_info,
                                  'cui2aui_id':cui2aui_id, 'unique': unique_id2aui,
                                  'aui2layer': ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.aui2layer_fn)), 'batch_size': FLAGS.batch_size,}
    #modifies input queue if we're only using pairs belonging to same semantic group
    if FLAGS.use_sg:
        intersect_sg2id = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.intersect_sg2id_pickle_fn))
        unique_sg2id = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.unique_sg2id_pickle_fn))
        for idx, id1 in enumerate(unique_id2aui.keys()):
            if (idx >= start_idx) and (idx <= end_idx):
                input_queue.put((aui_info[id1]["IDX"],id1))
        kwargs["intersect"] = intersect_sg2id
        kwargs["unique"] = unique_sg2id
    else:
        for idx, id1 in enumerate(unique_id2aui.keys()):
            if (idx >= start_idx) and (idx <= end_idx):
                input_queue.put((idx,id1))

    node_parallel = NodeParallel(generate_pairs_predict, None, n_processes, kwargs,
                                 None, False, logger = logger)
    node_parallel.set_input_queue(input_queue)
    node_parallel.run()
    del node_parallel
    del unique_id2aui
    del intersect_id2aui
    

def get_cui2aui_id_dict(mrconso_master_fn, cui_to_aui_id_pickle_fp,utils):
    cui_to_aui_dict = {}
    if os.path.isfile(cui_to_aui_id_pickle_fp):
        cui_to_aui_dict = utils.load_pickle(cui_to_aui_id_pickle_fp)
        return cui_to_aui_dict

    with open(mrconso_master_fn,'r') as fi:
        reader = csv.DictReader(fi, fieldnames = FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter,doublequote=False,quoting=csv.QUOTE_NONE)
        with tqdm(total = utils.count_lines(mrconso_master_fn)) as pbar:
            for line in reader:
                pbar.update(1)
                if line['CUI'] not in cui_to_aui_dict:
                    cui_to_aui_dict[line["CUI"]] = list()
                cui_to_aui_dict[line['CUI']].append(int(line['ID']))

    utils.dump_pickle(cui_to_aui_dict, cui_to_aui_id_pickle_fp)
    return cui_to_aui_dict


# def generate_negative_pairs(job_name, paths, prepy_cmds, submit_parameters):

#     start_time = time.time()

#     unique_id2aui = ruc.load_pickle(os.path.join(FLAGS.output_fp, FLAGS.unique_id2aui_pickle_fn))
#     #TODO: Modify swarm parameters to suit our use case
#     swarm_parameters = [
#         " --workspace_dp=%s"%FLAGS.workspace_dp,
#         " --umls_version_dp=%s"%FLAGS.umls_version_dp,
#         " --umls_dl_dp=%s"%FLAGS.umls_dl_dp,
#         " --n_processes=%d"%FLAGS.n_processes,
#         " --job_name=%s"%(job_name),
#         " --start_idx=start_index",
#         " --end_idx=end_index",
#         " --dataset_version_dn=%s"%(FLAGS.dataset_version_dn),
#         " --gen_master_file=false",
#         " --gen_pos_pairs=false",  
#         " --gen_neg_pairs=false",
#         " --gen_neg_pairs_batch=true",
#         " --neg_to_pos_rate=%d"%FLAGS.neg_to_pos_rate,
#         " --conda_env=%s"%FLAGS.conda_env,
#         " --debug=%s"%debug,
#         " --inputs_pickle_fp=%s"%paths['inputs_pickle_fp'],
#         " --completed_inputs_fp=%s"%(paths['completed_inputs_fp']),
#         " > %s"%(os.path.join(paths['data_gen_log_dp'], "%s_start_index_end_index.log "%(job_name))),
#         ]

#     # input_paras = {'inputs': paths['inputs_pickle_fp'],
#     #               'inputs_keys': paths['inputs_pickle_fp'] + FLAGS.inputs_keys_ext}

#     # output_paras = {'training_type': paths['neg_pairs_training_flavor_fp'], 
#     #                 'gentest_type': paths['neg_pairs_gentest_flavor_fp']}
#     # output_globs = {'training_type': paths['neg_pairs_training_flavor_glob'], 
#     #                 'gentest_type': paths['neg_pairs_gentest_flavor_glob']}

#     ninputs = len(unique_id2aui)
#     ntasks = 3 if FLAGS.debug is True else FLAGS.ntasks

#     # for k, fp in output_globs.items():
#     #     for flavor, flavor_glob in fp.items():
#     #         utils.clear(flavor_glob)

#     #takes in a random job name, a flag to actually run the job, commands that come before the python file, swarm_params that are a part of the python file, parameters for the .submit file, 
#     #TODO: set job_name to whatever we want
#     slurmjob = SlurmJob(job_name, FLAGS.run_slurm_job, prepy_cmds, swarm_parameters, submit_parameters, ntasks, ninputs, None, paths, logger=logger)
#     slurmjob.run()
    
def write_pairs_to_file(pairs_ds, pairs_fn):
    if len(pairs_ds) > 0:
        with open(pairs_fn,'a') as fo:
            for k in pairs_ds:                            
                fo.write(str(k["jacc"]) + FLAGS.delimiter)
                fo.write(k["AUI1"] + FLAGS.delimiter)
                fo.write(k["AUI2"] + FLAGS.delimiter + k['Label'] + "\n")

def write_list_to_file(pairs_ds, pairs_fn):
    with open(pairs_fn,'a') as fo:
            for k in pairs_ds: 
                fo.write(k)
                #fo.write('\n')
    return

def read_file_to_list(pairs_fp):
    pairs = []
    logger.info("Loading file %s ..."%pairs_fp)      
    with open(pairs_fp) as f:
        for line in f:
            pairs.append(line)  
    return pairs
                
def read_file_to_pairs(pairs_fp):
    pairs = []
    logger.info("Loading file %s ..."%pairs_fp)                
    with open(pairs_fp, 'r') as fi:
        reader = csv.DictReader(fi,fieldnames=["jacc", "AUI1", "AUI2", "Label"], delimiter=FLAGS.delimiter)
        with tqdm(total = utils.count_lines(pairs_fp)) as pbar:
            for line in reader:
                pbar.update(1)
                pairs.append(line)
    
    return pairs


        
def main(_):
    global utils
    utils = Utils()
    paths = dict()
    # Local folder, create if not existing
    # paths['log_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.log_dn)
    # Path(paths['log_dp']).mkdir(parents=True, exist_ok=True)
    # paths['data_gen_log_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.log_dn, FLAGS.job_name)
    # Path(paths['data_gen_log_dp']).mkdir(parents=True, exist_ok=True)

    # paths['extra_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.extra_dn)
    # Path(paths['extra_dp']).mkdir(parents=True, exist_ok=True)
    # # File path to SemGroups.txt
    # paths['sg_st_fp'] = os.path.join(paths['extra_dp'], FLAGS.sg_st_fn)

    # # For the executable files
    # paths['bin_dp'] = os.path.join(FLAGS.workspace_dp, FLAGS.bin_dn)
    # Path(paths['bin_dp']).mkdir(parents=True, exist_ok=True)
    
    # # Paths to METATHESAURUS filfes
    # paths['umls_meta_dp'] = FLAGS.umls_meta_dp if FLAGS.umls_meta_dp is not None else os.path.join(FLAGS.umls_version_dp, FLAGS.umls_meta_dn)
    
    # paths['mrxnw_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.mrxnw_fn)
    # paths['mrxns_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.mrxns_fn)
    # paths['mrconso_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.mrconso_fn)
    # paths['cui_sty_fp'] = os.path.join(paths['umls_meta_dp'], FLAGS.cui_sty_fn)

    
    # paths['datasets_dp'] = FLAGS.datasets_dp if FLAGS.datasets_dp is not None else os.path.join(FLAGS.workspace_dp, FLAGS.datasets_dn)
    # Path(paths['datasets_dp']).mkdir(parents=True, exist_ok=True)

    # # For the dataset files
    # paths['dataset_dn'] = get_dataset_dn()
    # paths['dataset_dp'] = os.path.join(paths['datasets_dp'], paths['dataset_dn'])
    # Path(paths['dataset_dp']).mkdir(parents=True, exist_ok=True)

    # paths['data_generator_fp'] = os.path.join(paths['bin_dp'], FLAGS.data_generator_fn)
    # paths['bin_umls_version_dp'] = os.path.join(paths['bin_dp'], paths['dataset_dn'])
    # Path(paths['bin_umls_version_dp']).mkdir(parents=True, exist_ok=True)
    # paths['swarm_fp'] = os.path.join(paths['bin_umls_version_dp'], "%s_%s"%(FLAGS.dataset_version_dn, FLAGS.swarm_fn))
    # paths['submit_gen_neg_pairs_jobs_fp'] = os.path.join(paths['bin_umls_version_dp'], "%s_%s"%(FLAGS.dataset_version_dn, FLAGS.submit_gen_neg_pairs_jobs_fn))
    

    # # File paths to program data files
    # paths['umls_dl_dp'] = FLAGS.umls_dl_dp if FLAGS.umls_dl_dp is not None else os.path.join(FLAGS.umls_version_dp, FLAGS.umls_dl_dn)
    # Path(paths['umls_dl_dp']).mkdir(parents=True, exist_ok=True)

    # paths['mrx_nw_id_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrx_nw_id_fn)
    # paths['mrx_ns_id_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrx_ns_id_fn)
    # paths['nw_id_aui_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.nw_id_aui_fn)    
    # paths['mrconso_master_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrconso_master_fn)  
    # paths['mrconso_master_randomized_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.mrconso_master_randomized_fn)  
    # paths['aui_info_gen_neg_pairs_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.aui_info_gen_neg_pairs_pickle_fn)
    # paths['cui_to_aui_id_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.cui_to_aui_id_pickle_fn)
    # paths['inputs_pickle_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.inputs_pickle_fn)

    # paths['pos_pairs_fp'] = os.path.join(paths['umls_dl_dp'], FLAGS.pos_pairs_fn)

    # paths['dataset_version_dp'] = os.path.join(FLAGS.umls_version_dp, FLAGS.dataset_version_dn)
    # Path(paths['dataset_version_dp']).mkdir(parents=True, exist_ok=True)

    # paths['neg_files_dp'] = os.path.join(paths['dataset_version_dp'], FLAGS.neg_file_prefix)
    # Path(paths['neg_files_dp']).mkdir(parents=True, exist_ok=True)
    # paths['neg_batch_files_dp'] = os.path.join(paths['dataset_version_dp'], FLAGS.neg_batch_file_prefix)
    # Path(paths['neg_batch_files_dp']).mkdir(parents=True, exist_ok=True)
        
    # paths['neg_pairs_flavors'] = [
    #     FLAGS.neg_pairs_flavor_topn_sim,
    #     FLAGS.neg_pairs_flavor_ran_sim,
    #     FLAGS.neg_pairs_flavor_ran_nosim,
    #     FLAGS.neg_pairs_flavor_all,
    #     FLAGS.neg_pairs_flavor_ran,
    # ]
    # # ========= FOR NEG BATCH ================
    # paths['neg_pairs_training_flavor_batch_fp'] = dict()
    # paths['neg_pairs_gentest_flavor_batch_fp'] = dict()
    
    # for flavor in paths['neg_pairs_flavors']:
    #     paths['neg_pairs_training_flavor_batch_fp'][flavor] = os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s_%d_%d.RRF"%(FLAGS.training_type,flavor, FLAGS.neg_batch_file_prefix, FLAGS.start_idx, FLAGS.end_idx)) 
    #     paths['neg_pairs_gentest_flavor_batch_fp'][flavor] = os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s_%d_%d.RRF"%(FLAGS.gentest_type, flavor, FLAGS.neg_batch_file_prefix, FLAGS.start_idx, FLAGS.end_idx)) 

    # # ====== FOR NEG GLOB ==================
    # paths['neg_pairs_training_flavor_glob'] = {flavor:os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s*"%(FLAGS.training_type, flavor, FLAGS.neg_batch_file_prefix)) for flavor in paths['neg_pairs_flavors']}
    # paths['neg_pairs_gentest_flavor_glob'] = {flavor:os.path.join(paths['neg_batch_files_dp'], "%s_%s_%s*"%(FLAGS.gentest_type, flavor, FLAGS.neg_batch_file_prefix)) for flavor in paths['neg_pairs_flavors']}

    # # ====== FOR COMPLETED_AUIS ==================
    # paths['completed_inputs_fp'] = os.path.join(paths['neg_batch_files_dp'], FLAGS.completed_inputs_fn)
    
    # # ========= FOR NEG FINAL FILES COLLECTED FROM BATCHES ================    
    # paths['neg_pairs_training_flavor_fp'] = dict()
    # paths['neg_pairs_gentest_flavor_fp'] = dict()
    # for flavor in paths['neg_pairs_flavors']:
    #     paths['neg_pairs_training_flavor_fp'][flavor] = os.path.join(paths['neg_files_dp'], "%s_%s_%s.RRF"%(FLAGS.training_type, flavor, FLAGS.neg_file_prefix))
    #     paths['neg_pairs_gentest_flavor_fp'][flavor] = os.path.join(paths['neg_files_dp'], "%s_%s_%s.RRF"%(FLAGS.gentest_type, flavor, FLAGS.neg_file_prefix))
    
    # # FOR THE FINAL DATASETS
    # paths['ds_training_flavor_dp'] = dict()
    # for flavor in paths['neg_pairs_flavors']:
    #     paths['ds_training_flavor_dp'][flavor] = os.path.join(paths['dataset_version_dp'], FLAGS.training_type, flavor)
    #     Path(paths['ds_training_flavor_dp'][flavor]).mkdir(parents=True, exist_ok=True)
        
    # paths['ds_gentest_flavor_dp'] = os.path.join(paths['dataset_version_dp'], FLAGS.gentest_type)
    # Path(paths['ds_gentest_flavor_dp']).mkdir(parents=True, exist_ok=True)
    
    # Logging

    if FLAGS.gen_neg_pairs:
        logger.info("Generating neg pairs files from %s ... "%(paths['mrconso_master_fp']))
        start_time = time.time()

        generate_negative_pairs(FLAGS.job_name, paths, prepy_cmds, submit_parameters)
        end_time = time.time()
        logger.info("Generating neg pairs in %d sec."%(end_time - start_time))

    if FLAGS.gen_neg_pairs_batch:
        generate_negative_pairs_node_batch(paths, FLAGS.start_idx, FLAGS.end_idx, FLAGS.neg_to_pos_rate, FLAGS.n_processes)
        
    logger.info("Finished.")

if __name__ == '__main__':
    app.run(main)

