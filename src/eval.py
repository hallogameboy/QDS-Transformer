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

    # Set up the experimental environment.
    exp = experiment.Experiment(FLAGS, cfg, dumpflag=False)

    for i, layer in enumerate(exp.model.base.encoder.layer):
        layer.attention.self.attention_window = FLAGS.window_size

    # Load the corpus.
    corpus = utils.Corpus(PATH_CORPUS, FLAGS)
    # Load train/dev data.
    test_data = utils.Data(PATH_DATA_PREFIX + 'test', corpus, FLAGS)
    

    # Evaluate dev data.
    test_eval = exp.eval_dump(test_data, FLAGS.num_sample_eval,
            'Evaluating test queries')
    print('Test Evaluation', test_eval, file=sys.stderr)


if __name__ == '__main__':
    app.run(main)

