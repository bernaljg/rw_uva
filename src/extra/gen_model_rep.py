import run_umls_classifier_2 as ruc
from absl import app, flags,logging
import os
FLAGS = flags.FLAGS
import os
import tensorflow as tf
import numpy as np
from math import ceil

def euc_test_dict(d):
    for i, (k,v) in enumerate(d.items()):
        if i > 2:
            break
        logging.info('key: {}, value: {}'.format(k,v))

def euc_gen_aui2vec(tokenizer, aui2vec_fp, mrc_atoms):
    """generates dictionary where key = aui_id and value = embedding

    Args:
        tokenizer (object): fitted tokenizer to generate embeddings
        aui2vec_fp (string): filepath to read or write aui2vec
        atom_pickle_fp (string): filepath of mrconso dictionary

    Returns:
        dict: key = aui, value = embedding 
    """
    aui2vec = dict()
    logging.info('checking for aui2vec_fp at {}'.format(aui2vec_fp))
    if os.path.isfile(aui2vec_fp):
        aui2vec = ruc.load_pickle(aui2vec_fp)
    else:
        aui2vec = ruc.gen_aui2vec(tokenizer, aui2vec_fp, mrc_atoms)
    return aui2vec


def gen_partition(aui2vec, val=False):
    """creates dictionary where key = (aui_id, aui_id) pairs value = arbitrary label
    For either creating aui2layer or evaluating aui2layer

    Args:
        aui2vec (dict): dictonary with aui_id as key and tokenized vector output as value
        val (bool): determines if pairs should consist of identical auis or a test set to evaluate aui2layer

    Returns:
        dict: key = tuple of auis, value = label 
    """
    auis = list(aui2vec.keys())
    partition = dict()
    if val == False:
        for aui_id in auis:
            partition[(aui_id,aui_id)] = np.random.randint(0,2)
        logging.info('testing parition...')
        euc_test_dict(partition)
    else:
        aui_id_1 = auis[0]
        for i in range(1,len(auis)):
            partition[(aui_id_1,auis[i])] = np.random.randint(0,2)
    return partition


# def gen_val_partition(ab_aui2vec):
#     """Generates dictionary where key = every possible pair that can be made from first aui in auis[], this serves to validate that we correctly extrated the 
#     correct n-1 layer representation for each aui string

#     Args:
#         ab_aui2vec (dict): key = aui, value = embedding

#     Returns:
#         dict: key = tuple of auis, value = label
#     """
    
#     partition = dict()
#     aui_id_1 = auis[0]
#     for i in range(1,len(auis)):
#         partition[(aui_id_1,auis[i])] = np.random.randint(0,2)
#         #partition[(aui_id_1, 8713194+436466+1+i)] = np.random.randint(0,2)
#     return partition

def get_model_output(generator, labels, labels_size, model, batch_size, epoch = None, log_scores = None):
    """generates model output using data generator

    Args:
        generator (object): datagenerator that allows us to predict on massive dataset
        labels_size (int): size of dataset
        model (object): model to get outputs from
        batch_size (int): batch size for predicting
        epoch (int, optional): # of epochs for training 
    """
  
    test_size = labels_size
    predict = model.predict(generator, 
#                             batch_size = FLAGS.batch_size, 
                            steps = ceil(test_size/batch_size),
                            use_multiprocessing = True,
                            workers = FLAGS.generator_workers,
                            verbose = FLAGS.predict_verbose)

   # print('shape of predict is: {}'.format(np.shape(predict)))
    return predict


def gen_aui2layer(layer_output, aui2layer_fp, aui2vec):
    """generates aui2layer dictionary where key = aui_id and value = n-1 layer output

    Args:
        layer_output (list): 2D matrix where each row reprents the n-1 layer output
        aui2layer_fp (string): filepath to read/write aui2layer dictionary
        aui2vec (dict): aui2vec dictionary

    Returns:
        dict: key = aui, value = n-1 layer representation of that aui
    """
    if(os.path.isfile(aui2layer_fp)):
        logging.info('fetching aui2layer at {}'.format(aui2layer_fp))
        return ruc.load_pickle(aui2layer_fp)
    aui2layer = dict()
    for idx,aui_id in enumerate(aui2vec.keys()):
        aui2layer[aui_id] = layer_output[idx]
    ruc.dump_pickle(aui2layer, aui2layer_fp)
    return aui2layer

def gen_model_rep(paths, logger):
    global logging
    logging = logger
    ab_mrc_atoms = ruc.load_pickle(paths['ab_mrc_atoms_fp'])
    tokenizer = ruc.load_pickle(paths['tokenizer_pickle_fp'])
    embedding_matrix = ruc.make_word_embeddings(embedding_filepath=paths['embedding_fp'], vocab_length=len(tokenizer.word_index) + 1, tokenizer=tokenizer)
    logging.info('embedding matrix size: {}'.format(np.shape(embedding_matrix)))
    model = ruc.create_model(embedding_matrix)
    model.load_weights(paths['checkpoint'])
    logging.info('length of tokenizer before: {}'.format(len(tokenizer.word_index) + 1))
    #tokenizer = gen_tokenizer(mod_tokenizer_fp, ulms_new_fp, unique_pickle_fp)
    #logging.info('length of tokenizer after: {}'.format(len(tokenizer.word_index) + 1))
    ab_aui2vec = euc_gen_aui2vec(tokenizer, paths['ab_aui2vec_fp'], ab_mrc_atoms) 
    logging.info('length of aui2vec: {}'.format(len(ab_aui2vec)))
    model_mod = tf.keras.Model(inputs = model.inputs, outputs = [model.layers[-1]._inbound_nodes[0].input_tensors])
    logging.info('getting layer output...')
    partition = gen_partition(ab_aui2vec)
    generator = ruc.DataGenerator(partition, ab_aui2vec, None, FLAGS.max_seq_length, 
                                               FLAGS.embedding_dim, FLAGS.word_embedding, FLAGS.context_vector_dim, FLAGS.exp_flavor,
                                               batch_size = FLAGS.batch_size, 
                                               shuffle = False, is_test = False)

    layer_output = get_model_output(generator, partition.values(), len(partition.values()), model_mod, FLAGS.batch_size, epoch = None, log_scores = None)
    #TODO: compare left and right for all auis 
    layer_output_left, layer_output_right = layer_output[0][0], layer_output[0][1]
    #logging.info('writing aui2layer...')
    aui2layer = gen_aui2layer(layer_output_left, paths['aui2layer_fp'], ab_aui2vec)



def main(argv):
    print(FLAGS.embedding_dim)
    print(FLAGS.aa_mrc_atoms_fn)
if __name__ == '__main__':
    app.run(main)