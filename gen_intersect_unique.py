import run_umls_classifier_2 as ruc
from tqdm import tqdm
import csv
from absl import app, flags,logging
import os
from collections import defaultdict
FLAGS = flags.FLAGS


def count_lines(filein):
    'Count total number of lines present in a file, helps with presenting correct loading bar via tqdm'
    return sum(1 for line in open(filein))

def gen_mrc_atom(mrc_atoms_fp, ulms_fp,id):
    print('in gen_mrc_atoms...')
    """Given mrconso_master file, converts to mrc_atoms dictionary where key = aui_id, value = CUI,LUI, SCUI,SG, STR, ID

    Args:
        mrc_atoms_fp (string): filepath where you want to store mrc_atoms dictionary
        ulms_fp (string): mrconso_master filepath
        id (int): integer to put as unique ID

    Returns:
        dict: key is AUI and value is dictionary containing CUI, LUI, SCUI, SG, STR, ID
        int: last unused integer for ID
    """
    if(os.path.isfile(mrc_atoms_fp)):
        logging.info('MRC_ATOMS found')
        return ruc.load_pickle(mrc_atoms_fp), id
    else:
        aui_dict = dict()
        logging.info('generating MRC_ATOM...')
        with open(ulms_fp, 'r') as f:
            reader = csv.DictReader(f, fieldnames=FLAGS.mrconso_master_fields, delimiter=FLAGS.delimiter)
            with tqdm(total = count_lines(ulms_fp)) as pbar:
                for idx,row in enumerate(reader):
                    pbar.update(1)
                    aui_dict[row['AUI']] = {'CUI': row['CUI'], 'LUI':row['LUI'], 'SCUI': row['SCUI'], 'SG':row['SG'], 'STR': row['STR'], 'ID': id}
                    id += 1

        #euc_test_dict(aui_dict) #save for unit testing
        ruc.dump_pickle(aui_dict, mrc_atoms_fp)
        return aui_dict, id
    
def compare(ab_mrc_atoms, aa_mrc_atoms, paths):
    """Takes in mrc_atoms dictionary and finds which auis intersect and which are unique to 2020AB, stores multiple dicts

    Args:
        aa_mrc_atoms (dict): mrc_atoms for UMLS version 2020AA
        ab_mrc_atoms (dict): mrc_atoms for UMLS version 2020AB
        intersect_id2aui_pickle_fp (string): filepath to write dictionary of auis present in both 2020AA and 2020AB, key = (int) id , value = aui id
        unique_id2aui_pickle_fp (string): filepath of dictionary of auis unique to 2020AB, key = (int) id, value = aui_id
        intersect_aui2id_pickle_fp (string): filepath of dictionary where key = aui_id, value = (int) id
        unique_aui2id_pickle_fp (string): filepath of dictionary of auis unique to 2020AB, key = aui_id, value = (int) id
    """
    intersect_id2aui = dict()
    unique_id2aui = dict()

    intersect_aui2id = dict()
    unique_aui2id = dict()
    
    intersect_sg2id = defaultdict(list)
    unique_sg2id = defaultdict(list)
    aui_info = dict()
    for aui, row in ab_mrc_atoms.items():
        temp_dict = {'AUI': aui, 'CUI': row['CUI'], "SG": row["SG"]}
        
        if aui in aa_mrc_atoms:
            intersect_id2aui[row['ID']] = aui
            intersect_aui2id[aui] = row['ID']
            sg = row["SG"].split(",")[0]
            if FLAGS.use_sg:
                intersect_sg2id[sg].append(row['ID'])
        else: 
            unique_id2aui[row['ID']] = aui
            unique_aui2id[aui] = row['ID']
            if FLAGS.use_sg:
                unique_sg2id[sg].append(row['ID'])
                temp_dict["IDX"] = len(unique_sg2id[sg]) - 1
        aui_info[row["ID"]] = temp_dict
        
    logging.info('generating intersect, unique_terms pickles...')
    ruc.dump_pickle(intersect_id2aui, paths['intersect_id2aui_pickle_fp'])
    ruc.dump_pickle(intersect_aui2id, paths['intersect_aui2id_pickle_fp'])
    ruc.dump_pickle(unique_id2aui, paths['unique_id2aui_pickle_fp'])
    ruc.dump_pickle(unique_aui2id, paths['unique_aui2id_pickle_fp'])
    ruc.dump_pickle(aui_info, paths['aui_info_fp'])
    if FLAGS.use_sg:
        ruc.dump_pickle(unique_sg2id, paths['unique_sg2id_pickle_fp'])
        ruc.dump_pickle(intersect_sg2id, paths['intersect_sg2id_pickle_fp'])
    
def gen_intersect_unique(paths, logger):
     global logging
     logging = logger
     ruc.logger = logger
     aa_mrc_atoms, end_id = gen_mrc_atom(paths['aa_mrc_atoms_fp'], paths['aa_umls_fp'],0)
     ab_mrc_atoms, _ = gen_mrc_atom(paths['ab_mrc_atoms_fp'], paths['ab_umls_fp'],end_id)
     compare(ab_mrc_atoms, aa_mrc_atoms, paths)
