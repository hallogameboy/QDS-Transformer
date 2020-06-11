#!/usr/bin/env python3
from absl import app, flags
import logging
import numpy as np
import random
import os
import sys
import yaml

import torch

import utils
import experiment

utils.handle_flags()



def main(argv):    
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    FLAGS = flags.FLAGS
    utils.print_flags(FLAGS)

    # Random seed initialization.
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)
    # Configuration and paths.
    cfg = yaml.load(open(FLAGS.config, 'r'), Loader=yaml.BaseLoader)
    PATH_DATA = cfg['path_data'] 
    PATH_CORPUS = '{}/{}'.format(PATH_DATA, cfg['corpus'])
    PATH_DATA_PREFIX = '{}/{}'.format(PATH_DATA, cfg['data_prefix'])
    PATH_MODEL_PREFIX = '{}/{}'.format(cfg['path_model'], FLAGS.model_prefix)
    os.makedirs(PATH_MODEL_PREFIX, exist_ok=True)

    # Set up the experimental environment.
    exp = experiment.Experiment(FLAGS, cfg)
    
    # Change attention window size.
    for i, layer in enumerate(exp.model.base.encoder.layer):
        layer.attention.self.attention_window = FLAGS.window_size

    # Load the corpus.
    corpus = utils.Corpus(PATH_CORPUS, FLAGS)
    # Load train/dev data.
    train_data = utils.Data(PATH_DATA_PREFIX + 'train', corpus, FLAGS)
    dev_data = utils.Data(PATH_DATA_PREFIX + 'dev', corpus, FLAGS)
    test_data = utils.Data(PATH_DATA_PREFIX + 'test', corpus, FLAGS)
    
    for epoch in range(FLAGS.last_epoch, FLAGS.num_epochs):
        print('Epoch {}'.format(epoch + 1), file=sys.stderr)
        # Train the model.
        train_loss = exp.train(train_data, 
                eval_data=dev_data,
                test_data=test_data,
                num_sample_eval=FLAGS.num_sample_eval)
        print('Epoch {}, train_loss = {}'.format(
            epoch + 1, train_loss), file=sys.stderr)

        # Dump the model.
        print('Dump model for epoch {}.'.format(epoch + 1))
        exp.dump_model(PATH_MODEL_PREFIX, str(epoch + 1))
 
        # Evaluate dev data.
        test_eval = exp.eval_dump(test_data, FLAGS.num_sample_eval,
                'Evaluating test queries')
        print('Test Evaluation', test_eval, file=sys.stderr)

        # Dump tensorboard results.
        if exp.tb:
            exp.tb_writer.add_scalar('Epoch_Eval_cut10/NDCG', test_eval['ndcg10'], epoch + 1)
            exp.tb_writer.add_scalar('Epoch_Eval_cut10/MRR', test_eval['mrr10'],  epoch + 1)
            exp.tb_writer.add_scalar('Epoch_Eval_cut10/MAP', test_eval['map10'], epoch + 1)
            exp.tb_writer.add_scalar('Epoch_Eval_overall/NDCG', test_eval['ndcg'], epoch + 1)
            exp.tb_writer.add_scalar('Epoch_Eval_overall/MRR', test_eval['mrr'], epoch + 1)
            exp.tb_writer.add_scalar('Epoch_Eval_overall/MAP', test_eval['map'], epoch + 1)


if __name__ == '__main__':
    app.run(main)

