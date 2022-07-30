import os
from pathlib import Path
from absl import app, flags,logging
import euc_run_data_generator as rdg
from euc_common import NodeParallel, Utils
FLAGS = flags.FLAGS
def predict_pairs(paths, logging):
    # paths = dict()

    # #For storing the logs 
    # paths['log_dp'] = os.path.join(FLAGS.output_fp, FLAGS.log_dn)
    # Path(paths['log_dp']).mkdir(parents=True, exist_ok=True)
    # paths['log_filepath'] = os.path.join(paths['log_dp'],"%s.log"%(FLAGS.application_name))
    utils = Utils()
    logger = utils.get_logger(logging.INFO, FLAGS.application_name, paths['log_filepath'])
    utils.set_logger(logger)
    rdg.set_utils(utils)

    # #For the executable files
    #paths['output_fp'] = FLAGS.output_fp
    # Path(paths['bin_dp']).mkdir(parents=True, exist_ok=True)

    #generating neccessary files 
    aui_info = rdg.get_aui_info_gen_neg_pairs( os.path.join(FLAGS.output_fp, FLAGS.ab_mrc_atoms_fn) , os.path.join(FLAGS.output_fp, FLAGS.aui_info_fn), utils)
    
    #cui2aui_id = rdg.get_cui2aui_id_dict(os.path.join(FLAGS.umls_fp, FLAGS.ab_umls_fn + "/META_DL/MRCONSO_MASTER.RRF"), os.path.join(FLAGS.output_fp, FLAGS.cui2aui_id_fn),utils)
  
    cui2aui_id = dict()

    if FLAGS.parallelize == True:
         rdg.parallel_predict_pairs(FLAGS.job_name, paths,logger)
        
    else:
        rdg.gen_pairs_predict_batch(FLAGS.start_idx, FLAGS.end_idx, aui_info, cui2aui_id, logger, FLAGS.n_processes)
   
