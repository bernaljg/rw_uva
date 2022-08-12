from multiprocessing import Manager, Queue, Process, Lock
from pathlib import Path
import subprocess
import logging
import pickle
import os
import inspect
import math
import glob
import time
import random
import csv
import gc
import numpy
from tqdm import tqdm
#sys.path.append('/data/Bodenreider_UMLS_DL/MedInfo2021')
from common import NodeParallel as ruc_NodeParallel


class NodeParallel(ruc_NodeParallel):

    def __init__(self, worker_target, output_target, num_processes, worker_target_kwargs, output_target_kwargs, kill_process, logger=None):
        super().__init__(worker_target, output_target, num_processes, {}, worker_target_kwargs, output_target_kwargs, logger = logger)
        self.completed = dict()
        self.completed['queue'] = self.manager.Queue()
        self.completed['fp'] = 'completed.txt'
        self.completed['status'] = False
        self.kill_process = kill_process
        

    def create(self):
        '''Creates single process for output target and multiple processes for worker target'''
        for idx in range(self.num_workers):
            print('creating process for worker')
            p = Process(target = self.worker_target, args=(self.input_queue, self.being_processed,self.completed,), kwargs = self.worker_kwargs)
            self.processes.append(p)
            self.worker_processes.append(p)
     
    def kill_ran_process(self):
        time.sleep(25)
        p = random.choice(self.processes)
        self.logger.info('KLLING PROCESS {}'.format(p.pid))
        os.kill(p.pid,9)
    
    def start(self):
        if len(self.processes) == 0:
            self.create()
        for p in self.processes:
            p.start()
        
        for p in self.worker_processes:
            self.worker_id_processes[p.pid] = p
        if self.kill_process == True:
            self.kill_ran_process()
        return

    def is_done(self):
        if (self.is_being_processed_done() is False) or (self.is_input_done() is False) or (self.is_completed_done() is False):
            return False
        return True

    def processing(self):
        '''Restarts worker in case of unexpected job exit'''
        num_completed = 0
        # completed_fo = open(self.completed['fp'],'w')
        # completed_fo.close()       
	#self.logger.debug("Remove files %s*"%(para['fp']))
            #clear(para['fp'])
        while (self.is_done() is False):
            for (idx,aui_id), p_id in self.being_processed.items():
                if self.worker_id_processes[p_id].is_alive() is False:

                    self.input_queue.put((idx,aui_id))
                    del self.being_processed[(idx,aui_id)]
                    #self.logger.info("Process %s is terminated by exception while processing %s."%(p_id, aui_id))
                    print("Process %s is terminated by exception while processing %s."%(p_id, aui_id))

                    del self.worker_id_processes[p_id]

                    # Add a new process
                    p =  Process(target = self.worker_target, args=(self.input_queue, self.being_processed,self.completed,), kwargs = self.worker_kwargs)
                    self.processes.append(p)
                    self.worker_processes.append(p)
                    p.start()

                    time.sleep(10)
                    self.worker_id_processes[p.pid] = p
                    #self.logger.info("Adding process %s"%p.pid)
                    print("Adding process %s".format(p.pid))
                    
            num_completed = self.track(num_completed)
            self.logger.info("Inputs completed: %d"%num_completed)
            time.sleep(10)

       # completed_fo.close()
        self.logger.info('Size of completed queue: {}'.format(self.completed['queue'].qsize()))
        self.logger.info("Finished with %d inputs"%num_completed)
        return

    def set_input_queue(self, input_queue):
        self.ninputs = input_queue.qsize()
        self.logger.info('size of input queue = {}'.format(input_queue.qsize()))
        while (input_queue.empty() is False):
            idx,aui_id = input_queue.get()
            self.input_queue.put((idx,aui_id))
        # for idx in range(self.num_workers):
        #     self.input_queue.put("END")

        return
        
class Utils:

    def __init__(self, logger = None):
        self.logger = logger

    def randomize_keys(self, d_fp):
        d = self.load_pickle(d_fp)
        keys = [k for k in d.keys()]
        random.shuffle(keys)
        random.shuffle(keys)
        random.shuffle(keys)
        self.dump_pickle(keys, d_fp + '_keys')
 
        return

    def merge_clusters(self, ori_fp, fp, final_fp):
        ori_dict = self.load_pickle(ori_fp)
        if (os.path.isfile(final_dict)):
            final_dict = self.load_pickle(final_fp)
        else:
            final_dict = dict()

        merged_dict = self.load_pickle(fp)
        merged_dict_keys = [k for k in merged_dict.keys()]

        new_merged_dict = dict()
        cnt = 0
        while cnt < len(merged_dict_keys):
            k1 = merged_dict_keys[cnt]
            k1_mergeable_k2 = merged_dict[k1]
            if len(k1_mergeable_k2) > 0:
                new_cluster = dict()

                # get all clusters in k1_mergeable_k2 
                for k2 in k1_mergeable_k2:
                    # recursively find all mergeable k2_mergeable_k3
                    new_cluster = new_cluster.union(merged_dict[k2])

                # generate the new key
                new_cluster_ids = list(k1_mergeable_k2.union({k1}))
                new_cluster_ids.sort()
                new_cluster_key = '_'.join(new_cluster_ids)
               
                # update merged_dict with the new key replacing all keys
            
    
                new_merged_dict[new_cluster_key] = new_cluster

            cnt += 1
        self.dump_pickle(new_merged_dict, fp)
        return    

    def process_union(self,fp):
        d = self.load_pickle(fp)
        out = dict()
        for k, lst in d.items():
            union_all = set()
            for cluster in lst: # list of items
                union_all = union_all.union(set(cluster))
            out[k] = union_all

        self.dump_pickle(out, fp)
        del d
        del out
        return

    def collect(self, okey, okey_glob, fp): 
        'Collect list of items from write()'

        #Collect the results from output_paras
        self.logger.debug("Started collecting results for %s from %s"%(okey, fp + '*'))
        output_dict = dict()
        output_files = glob.glob(okey_glob + '*')
        num_updates = 0
        repeated_keys = 0
        for f in output_files:
            d = self.load_pickle(f)
            num_updates += 1
            for k, item_lst in d.items():
                if k not in output_dict:
                    output_dict[k] = list()
                else:
                    repeated_keys += 1
                for item in item_lst:
               	    output_dict[k].append(item)
            del d
        self.logger.debug("Collected %d keys (%d repeated, %d updates) for %s from %s"%(len(output_dict), repeated_keys, num_updates, okey, fp + '*'))
        self.dump_pickle(output_dict, fp)
        del output_dict

        return

    def merge(self, okey, okey_glob, fp):
        'Merge items from glob' 
        self.logger.debug("Started merging results for %s from %s"%(okey, fp + '*'))
        output_dict = dict()
        output_files = glob.glob(okey_glob + '*')
        num_updates = 0
        repeated_keys = 0
        for f in output_files:
            d = self.load_pickle(f)
            num_updates += 1
            for k, item in d.items():
                if k not in output_dict:
                    output_dict[k] = list()
                else:
                    repeated_keys += 1
                output_dict[k].append(item)
            del d
        self.logger.debug("Merged %d keys (%d repeated, %d updates) for %s from %s"%(len(output_dict), repeated_keys, num_updates, okey, fp + '*'))
        self.dump_pickle(output_dict, fp)
        del output_dict

        return

    def read_file_to_id_ds(self, fp, aui2id):
        pickle_fp = fp + '.PICKLE'
        partition = dict()
        if (os.path.isfile(pickle_fp)):
            self.logger.debug("Loading file %s ..."%pickle_fp)
            partition = self.load_pickle(pickle_fp)
            #return partition
        else:
            self.logger.debug("Loading file %s ..."%fp)

            partition = dict()
            with open(fp, 'r') as fi:
                reader = csv.DictReader(fi, fieldnames = ["jacc", "AUI1", "AUI2", "Label"], delimiter = '|')
                with tqdm(total = self.count_lines(fp)) as pbar:
                    for line in reader:
                        pbar.update(1)
                        #if ((aui2id[line['AUI1']] in aui2vec) and (aui2id[line['AUI2']] in aui2vec)):
                        ID = (aui2id[line['AUI1']], aui2id[line['AUI2']])
                        partition[ID] = (float(line['jacc']), int(line['Label']))

            self.dump_pickle(partition, pickle_fp)
        return partition                  

    def compute_scores(self, y_true, y_pred):
        TP, TN, FP, FN = 0, 0, 0, 0
        for t, p in zip(y_true, y_pred):
            if t == 1 and p == 1:
                TP += 1
            elif t == 1 and p == 0:
                FN += 1
            elif t == 0 and p == 1:
                FP += 1
            else:
                TN += 1
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        precision, recall, f1 = 0, 0, 0

        if TP + FP > 0:
            precision = TP/(TP+FP)

        if TP + FN > 0:
            recall = TP/(TP+FN)

        if recall + precision > 0:
            f1 = 2*(recall * precision) / (recall + precision)

        return round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)

    def cal_scores(self, TP, TN, FP, FN):
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        precision, recall, f1 = 0, 0, 0

        if TP + FP > 0:
            precision = TP/(TP+FP)

        if TP + FN > 0:
            recall = TP/(TP+FN)

        if recall + precision > 0:
            f1 = 2*(recall * precision) / (recall + precision)

        return round(accuracy,4), round(precision,4), round(recall,4), round(f1,4)

    def clear(self, path):
        fps = glob.glob(path + '*')
        for fp in fps:
            f = Path(fp)
            f.unlink()

    def count_lines(self, filein):
        return sum(1 for line in open(filein))
   
    def set_logger(self, logger):
        self.logger = logger

    def get_logger(self, log_level, name, filepath):
        # get TF logger
    
        log = logging.getLogger(name)
        log.setLevel(log_level)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
        ch = logging.StreamHandler()
        ch.setLevel(level=logging.INFO)
        ch.setFormatter(formatter)
        
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filepath)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
    
        log.addHandler(ch)
        log.addHandler(fh) 
        return log

    def get_important_logger(self, log_level, name, filepath):
        # get TF logger

        log = logging.getLogger(name)
        log.setLevel(log_level)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # create file handler which logs even debug messages
        fh = logging.FileHandler(filepath)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)

        log.addHandler(fh)
        return log

    def dump_pickle(self, obj, pickle_fp):
        self.logger.debug("Dumping pickle at %s"%pickle_fp)
        with open(pickle_fp, 'wb') as f:
            pickle.dump(obj, f, protocol = 4)

        if type(obj) == list:
            self.test_list(obj)
        elif type(obj) == dict:
            self.test_dict(obj)
        return obj

    def load_pickle(self, pickle_fp):
        self.logger.debug("Loading %s"%pickle_fp)
        with open(pickle_fp, 'rb') as f:
            obj = pickle.load(f)
        if type(obj) == list:
            self.test_list(obj)
        elif type(obj) == dict:
            self.test_dict(obj)
        return obj

    def test_big_item(self, d, n):
        cnt = 0
        for k, v in d.items():
            if len(v) > n:
                cnt += 1
                self.logger.debug("Big AUI: item has len > {}: [{}] = {} " %(n, id1, len(id2s)))
        return 

    @staticmethod
    def test_dict(d, dn=None):
        for i, (k,v) in enumerate(d.items()):
            if i > 2:
                break
            logging.info('key: {}, value: {}'.format(k,v))

    def test_list(self, l, dn=None):
        for i, v in enumerate(l):
            if i < 2:
                self.logger.debug('{}({}): [{}] = {}'.format(inspect.stack()[1][3], dn, i, v))
        return

    def test_type(self, t, dn=None):
        self.logger.debug('{}({}): type({}) = {}'.format(inspect.stack()[1][3],dn, t, type(t)))
    
    def test_member(self, e, d, dn=None):
        if e not in d:
            self.logger.debug('{}({}): {} not in {}'.format(inspect.stack()[1][3], dn, e, d))
        else:
            self.logger.debug('{}({}): t[{}] = {}'.format(inspect.stack()[1][3], dn, e, d[e]))

    def shuffle_file(self, file_in, file_out):
        lines = open(file_in).readlines()
        random.shuffle(lines)
        open(file_out, 'w').writelines(lines)
        return


class SlurmJob():
    def __init__(self, job_name, run_slurm_job, prepy_cmds, swarm_parameters, submit_parameters, ntasks, ninputs, fp_suffix, paths, job_id = None, max_concurrent = 1000, logger = None):
        self.prepy_cmds = prepy_cmds # source /data/nguyenvt2/libs/miniconda3/etc/profile.d/conda.sh; conda activate {};
                                    # --module=python/3.7
        self.execute_py_fp = paths['execute_py_fp']
        self.swarm_parameters = swarm_parameters
        self.submit_parameters = submit_parameters
        #self.bin_dp = paths['bin_dp']
        self.job_name = job_name
        self.run_slurm_job = run_slurm_job
        self.ntasks = ntasks
        self.ninputs = ninputs
        self.fp_suffix = fp_suffix if fp_suffix is not None else ''
        # self.input_paras = input_paras
        # self.output_paras = output_paras
        # self.output_globs = output_globs
        self.max_concurrent = max_concurrent
        self.logger = logger
        # self.utils = Utils()
        # self.utils.set_logger(logger)
      
        self.swarm_fp = os.path.join(paths['output_fp'], self.job_name + '.swarm' + self.fp_suffix)
        self.submit_fp = os.path.join(paths['output_fp'], self.job_name + '.submit' + self.fp_suffix)

        self.swarm_files = list()
        self.submit_files = list()
    
        self.start_time = time.time()
        self.time_limit = 10*20*60*60

    def run(self, submit=True):
        'Generate swarm files and submit_job files'
       
        self.gen()

        #'Remove existing files'
        #for okey, fp in self.output_paras.items():
            #self.logger.debug("Deleting files %s*"%(fp))
            #clear(fp)
            #self.logger.debug("Deleting files %s*"%(self.output_globs[okey]))
            #clear(self.output_globs[okey])

        'Execute submit files' 
        if (self.run_slurm_job is True):
            if (self.check() is True):
                for f in self.submit_files:
                    self.logger.info("Excuting %s with slurm"%f)
                    self.execute_task(f)
            self.logger.info("Waiting for the job %s"%self.job_name)
            self.wait(self.job_name, 10)
        else:
            'Execute swarm files'
            for f in self.swarm_files:
                self.logger.debug("Executing %s with python"%f)
                self.execute_task(f)

    # def collect(self):
    #     'Collect results and pickle them into output_paras' 
    #     for okey, fp in self.output_paras.items():      
    #         self.utils.collect(okey, self.output_globs[okey], fp)


    def execute_task(self, fp):
        'Executing the task'
        subprocess.run(['cat','%s'%fp], check=True)
        subprocess.run(['sh','%s'%fp], check=True)
        return 
    

    def wait(self, job_name, interval = 10, job_id = None):
        
        time.sleep(interval)
        t = time.time()
        while True:
            # Break if this takes longer than time limit
            #if (time.time() - t > time_limit):
            #    self.resume_jobs = True
            #    break
            # Check if the job is done
            if (self.check(job_name, job_id)):
                break
            time.sleep(interval)
        return

    def check(self, job_name = None, job_id=None):
        # Greb the jobs using sjobs
        if job_id is not None:
            chk_cmd = 'sjobs | grep %s | wc -l '%(job_id)
        else:
            if job_name is None:
                job_name = self.job_name
            grep_str = '%s'%(job_name)
            chk_cmd = 'sjobs | grep %s | wc -l '%(grep_str[0:8])

        chk_cmd_response = subprocess.getoutput(chk_cmd)
        self.logger.debug(chk_cmd_response)
        
        if int(chk_cmd_response.strip()) == 0:
            return True
        return False

    # def delete(self, swarm_fp = None, submit_fp = None):
    #     fps = glob.glob(os.path.join(self.bin_dp, self.job_name + "*.swarm"))
    #     fps += glob.glob(os.path.join(self.bin_dp, self.job_name + "*.sh"))
    #     for fp in fps:
    #         f = Path(fp)
    #         f.unlink()

    def gen(self, swarm_fp = None, submit_fp = None, override = False):
        if len(self.submit_files) > 0:
            if not override:
                return self.submit_files
        batch_size = math.ceil(self.ninputs/self.ntasks) #batch size is based off of the number of nodes and number of auis in unique set
        cnt = 0
        cur_file = 0
        num_files = self.ntasks/self.max_concurrent

        self.swarm_files = [] #list of filepaths 
        self.submit_files = [] #list of submit paths, seperated by cnt
        swarm_fp = swarm_fp if swarm_fp is not None else self.swarm_fp
        submit_fp = submit_fp if submit_fp is not None else self.submit_fp
        for i in range(0, self.ntasks):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size - 1
            if end_idx >= self.ninputs:
                end_idx = self.ninputs-1 #inclusive end_idx

            #append all swarm parameters (modified or not) to new_swarm_parameters
            new_swarm_parameters = []
            for idx in range(len(self.swarm_parameters)):
                    new_swarm_parameters.append(self.swarm_parameters[idx].replace('start_index', str(start_idx)).replace('end_index', str(end_idx)))

            if cnt == 0:

                #generate another set of swarm and submit files
                new_swarm_fp = swarm_fp + "_" + str(cur_file) #modified name with the index of the parent job
                self.swarm_files.append(new_swarm_fp)
                new_submit_fp = submit_fp + "_" + str(cur_file)
                self.submit_files.append(new_submit_fp)

                #write to swarm file
                with open(new_swarm_fp,'w') as fo1:
                    fo1.write("#!/bin/bash\n")        
                    fo1.write("{}; python {} ".format("; ".join(self.prepy_cmds), self.execute_py_fp))
                    fo1.write(" ".join(new_swarm_parameters))
                    fo1.write("\n")

                #write to submit file, we only want to write one and it will send every job that we want
                with open(new_submit_fp,'w') as fo2:
                    fo2.write("swarm -f " + new_swarm_fp)
                    fo2.write(" --job-name=%s"%self.job_name)
                    fo2.write(" ".join(self.submit_parameters))
                    fo2.write("\n")
            else:
                with open(new_swarm_fp,'a') as fo3:
                    fo3.write("{}; python {}".format("; ".join(self.prepy_cmds), self.execute_py_fp))
                    fo3.write(" ".join(new_swarm_parameters))
                    fo3.write("\n")
            cnt += 1
            if cnt == self.max_concurrent:
                cnt = 0
                cur_file += 1 #allows us to create another .submit file that points to another set of jobs within swarm file
        
        return self.swarm_files, self.submit_files