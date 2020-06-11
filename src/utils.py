from absl import flags
import numpy as np
import sys
import subprocess
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from collections import defaultdict
import random
import pytrec_eval
try:
    import ujson as json
except:
    import json

import torch
import torch.nn as nn
from transformers import RobertaTokenizer

import transformers 


def handle_flags():
    # Data configuration.
    flags.DEFINE_string('config',
            'config.yml', 'configure file (default: config.yml)')
    flags.DEFINE_string('model_prefix',
            'model', 'the prefix name of model files (default: model)')
 
    # Model parameters.
    flags.DEFINE_string('model',
            'QDST', 'the applied model (default: QDST')
    flags.DEFINE_string('load_model', '',
            'model path for continuing training (default: "" for not loading)')

    ## General parameters.
    flags.DEFINE_integer('max_seq_len',
            2048, 'Max length of each query + doc  sequence (default: 4096)')
    flags.DEFINE_integer('max_sent_num',
            256, 'Max number of sentences (default: 256)')
    flags.DEFINE_integer('hidden_dim',
            128, 'Hidden dimension (default: 128)')
    flags.DEFINE_string('hidden_func',
            'relu6', 'Non-linear function for hidden layers (default: relu6)')

    # Analysis parameters.
    flags.DEFINE_integer('window_size',
            64, 'Half window size for local attention')

    # Training parameters.
    flags.DEFINE_string('pred_file', None, 'path of prediction')
    flags.DEFINE_integer('last_epoch',
            0, 'Last epoch for the loaded model (default: 0)')
    flags.DEFINE_integer('batch_size', 1, 'Batch Size (default: 1)')
    flags.DEFINE_integer('num_epochs',
            10, 'Number of training epochs (default: 10)')
    flags.DEFINE_integer('random_seed',
            252, 'Random seeds for reproducibility (default: 252)')
    flags.DEFINE_integer('num_neg_samples',
            1, 'Number of negative examples for each positive one (default: 1)')
    flags.DEFINE_float('learning_rate',
            1e-5, 'Learning rate while training (default: 1e-5)')
    flags.DEFINE_float('l2_reg',
            1e-3, 'L2 regularization lambda (default: 1e-3)')


    # Attention parameters.
    flags.DEFINE_integer('attn_pattern',
            2, 'Attention pattern, ' + \
                    '0: Sparse-Transformer, ' + \
                    '1: Longformer-QA, ' + \
                    '2: QDS-Transformer, ' + \
                    '3: QDS-Transformer (S) ' + \
                    '4: QDS-Transformer (Q) ' + \
                    '(default: 2)')

    # Tensorboard parameters.
    flags.DEFINE_boolean('tb',
            False, 'Whether to write tensorboard logs.')
    flags.DEFINE_integer('tb_per_iter',
            3000, 'Per number of iterations to write tensorboard logs.')

    # Evaluation setting.
    flags.DEFINE_integer('num_sample_eval', 0,
            '# of sampled queries for eval (default: 0, <=0 for not sampling')
    FLAGS = flags.FLAGS


def print_flags(FLAGS):
    flag_dict = FLAGS.flag_values_dict() 
    print('===== FLAGS =====', file=sys.stderr)
    for f in flag_dict:
        print('{}: {}'.format(f, flag_dict[f]), file=sys.stderr)

    print('=================', file=sys.stderr)


def get_line_number(file_name):
    return int(subprocess.check_output(
        'wc -l {}'.format(file_name), shell=True).split()[0])

def err_measure(ranking, max = 10, max_grade=2):
    if max is None:
        max = len(ranking)
    
    ranking = ranking[:min(len(ranking), max)]
    ranking = map(float, ranking)

    result = 0.0
    prob_step_down = 1.0
    
    for rank, rel in enumerate(ranking):
        rank += 1
        utility = (pow(2, rel) - 1) / pow(2, max_grade)
        result += prob_step_down * utility / rank
        prob_step_down *= (1 - utility) 
      
    return result


class Evaluater:
    def __init__(self):
        self.reset()

    def reset(self):
        self.err_list = []
        self.err20_list = []
        self.rr_list = []
        self.rr10_list = []
        self.qrel_list = []
        self.run_list = []
        self.q_list = []

    def summary(self):
        qrel = {}
        run = {}
        assert(len(self.qrel_list) == len(self.run_list))
        for i in range(len(self.qrel_list)):
            assert(len(self.qrel_list[i]) == len(self.run_list[i]))
            qid = 'q{}'.format(i + 1)
            qrel[qid] = {}
            run[qid] = {}
            for j in range(len(self.run_list[i])):
                did = 'd{}'.format(j + 1)
                qrel[qid][did] = int(self.qrel_list[i][j])
                run[qid][did] = float(self.run_list[i][j])
        

        evaluater = pytrec_eval.RelevanceEvaluator(
                qrel, {'map', 'map_cut', 'ndcg', 'ndcg_cut', 'recall'}) 
        trec = evaluater.evaluate(run)
        results = {
                'mrr': np.mean(self.rr_list),
                'mrr10': np.mean(self.rr10_list),
                'err': np.mean(self.err_list),
                'err20': np.mean(self.err20_list),
                'map': np.mean([trec[d]['map'] for d in trec]),
                'map10': np.mean([trec[d]['map_cut_10'] for d in trec]),
                'map20': np.mean([trec[d]['map_cut_20'] for d in trec]),
                'ndcg': np.mean([trec[d]['ndcg'] for d in trec]),
                'ndcg10': np.mean([trec[d]['ndcg_cut_10'] for d in trec]),
                'ndcg20': np.mean([trec[d]['ndcg_cut_20'] for d in trec]),
                'recall100': np.mean([trec[d]['recall_100'] for d in trec])}
        return results

    def eval(self, y_pred, y_true, q=None):
        assert(len(y_pred) == len(y_true))
        self.qrel_list.append(y_true)
        self.run_list.append(y_pred)
        if q != None:
            self.q_list.append(q)

        ranked = sorted(list(zip(y_pred, y_true)), key=lambda x: -x[0])
        rr = -1
        rr10 = -1
        for i in range(len(ranked)):
            yp, yt = ranked[i]
            if yt > 0:
                if rr < 0:
                    rr = 1.0 / float(i + 1)
                    if i < 10: rr10 = rr
        if rr < 0: rr = 0.0
        if rr10 < 0: rr10 = 0.0

        self.rr_list.append(rr)
        self.rr10_list.append(rr10)

        ranking = [x[1] for x in ranked]
        self.err_list.append(err_measure(ranking, max=None, max_grade=4))
        self.err20_list.append(err_measure(ranking, max=20, max_grade=4))


class Corpus:
    def __init__(self, file_name, FLAGS):
        self.read_file(file_name)

    def read_file(self, file_name):
        self.corpus = {}
        # Read the corpus from the file.
        # tsv: docid, url, title, body.
        lc = get_line_number(file_name)
        with open(file_name, 'r') as fp:
            for line in tqdm(fp, 'Load {}'.format(file_name), total=lc):
                data = json.loads(line)
                # Dump into corpus.
                self.corpus[data['docid']] = data['ss']
        self.doc_list = list(self.corpus.keys())

    def get(self, docid):
        if docid in self.corpus:
            return self.corpus[docid]
        else:
            return self.sample()

    def sample(self):
        return self.get(random.choice(self.doc_list))


class Data:
    def __init__(self, file_prefix, corpus, FLAGS):
        # Data-related hyperparameters.
        self.max_seq_len = FLAGS.max_seq_len
        self.max_sent_num = FLAGS.max_sent_num
        self.batch_size = FLAGS.batch_size
        self.num_neg_samples = FLAGS.num_neg_samples
        self.window_size = FLAGS.window_size

        self.attn_cnt = 0.0
        self.attn_total = 0.0

        # Model-related hyperparameters.
        self.attn_pattern = FLAGS.attn_pattern
        assert(self.attn_pattern in [0, 1, 2, 3, 4, 5])

        # Tokenizer.
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

        # Objects.
        self.corpus = corpus
        self.query = {}
        self.qrels = defaultdict(int)
        self.qrels_by_q = defaultdict(list)
        self.qd_bm25 = {}
        self.qrels_qid = set()
        self.pos_records = []
        self.neg_records = []
        self.neg_lookup = defaultdict(list)
        self.candidates = defaultdict(list)

        # Read qrels.
        qrels_file = file_prefix + '-qrels.tsv'
        lc = get_line_number(qrels_file)
        with open(qrels_file, 'r') as fp:
            for line in tqdm(fp, desc='Load {}'.format(qrels_file), total=lc):
                qid, _, docid, r = line.split()
                assert((qid, docid) not in self.qrels)
                self.qrels[(qid, docid)] = int(r)
                self.qrels_by_q[qid].append((docid, int(r)))
                self.qrels_qid.add(qid)

        # Read queries.
        query_file = file_prefix + '-queries.tsv'
        lc = get_line_number(query_file)
        with open(query_file, 'r') as fp:
            for line in tqdm(fp, desc='Load {}'.format(query_file), total=lc):
                qid, q = line.split('\t')
                if qid not in self.qrels_qid: continue
                self.query[qid.strip()] = q.strip()

        # Read top-100 candidates.
        can_file = file_prefix + '-top100'
        lc = get_line_number(can_file)
        with open(can_file, 'r') as fp:
            for line in tqdm(fp, desc='Load {}'.format(can_file), total=lc):
                qid, _, docid, _, bm25, _ = line.split()
                if qid not in self.query: continue
                self.candidates[qid].append(docid)
                self.qd_bm25[(qid, docid)] = float(bm25)
                if self.qrels[(qid, docid)] > 0:
                    self.pos_records.append((qid, docid))
                else:
                    self.neg_records.append((qid, docid))
                    self.neg_lookup[qid].append(docid)

        for qid in self.candidates:
            self.candidates[qid] = self.candidates[qid][:100]


        # Validation.
        to_del = []
        for qid in self.query:
            if qid not in self.candidates:
                print('{} does not have candidates'.format(qid))
                to_del.append(qid)
        for x in to_del:
            del self.query[qid]
        
    def get_qid_list(self, num_sample_eval):
        if num_sample_eval <= 0:
            return list(self.query.keys())
        else:
            qids = list(self.query.keys())
            random.shuffle(qids)
            return qids[:num_sample_eval]

    def pad(self, x):
        y = np.zeros(self.L * self.max_len, dtype=np.int32)
        RL = min(len(x), self.L * self.max_len)
        y[:RL] = x[:RL]
        return y

    def get_inputs(self, qid, docid, masked_lm=False):
        query_text = self.query[qid]
        sents = self.corpus.get(docid)

        # Query tokens.
        if masked_lm:
            input_ids = []
            QL = 0
        else:
            # Query tokens.
            input_ids = [0] # Global attention token.
            input_ids.extend(self.tokenizer.encode(query_text))
            QL = len(input_ids)

        # Sentence tokens.
        # input_ids.append(0) # Global attention token.
        sid = 0
        sent_locs = []
        sent_mask = []
        while sid < self.max_sent_num and len(input_ids) < self.max_seq_len:
            sent_locs.append(len(input_ids))
            sent_mask.append([1.0])
            # Add the global attention token and sent tokens.
            input_ids.append(0)
            input_ids.extend(self.tokenizer.encode(sents[sid]))
            sid += 1
            if sid == len(sents): break
        last_pos = len(input_ids)
        input_ids.append(2)
        
        # Padding handling.
        L = len(input_ids)
        num_sent = len(sent_locs)
        if self.batch_size > 1:
            if num_sent < self.max_sent_num:
                sent_mask.extend(
                        [[0.] for i in range(self.max_sent_num - num_sent)])
                sent_locs.extend([0] * (self.max_sent_num - num_sent))
            if L < self.max_seq_len:
                input_ids.extend([1] * (self.max_seq_len - 1 - L))
                input_ids.append(2)

        if L > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            L = self.max_seq_len

        if len(input_ids) < 16:
            input_ids.extend([1] * (16 - L))


        ## Fill the attention masks.
        # Anchor BOS indices.
        bos_idx = [i for i, x in enumerate(input_ids) if x == 0]
        # Fill in local attention first.
        tok_mask = np.zeros([len(input_ids)], dtype=np.long)
        tok_mask[range(L)] = 1

        # Adjust global attention by the pattern.
        # Longformer-QA
        if self.attn_pattern == 1:
            tok_mask[0] = 2
            tok_mask[sent_locs[0]] = 2
            self.attn_cnt += 2


        # QDS-Transformer
        if self.attn_pattern == 2:
            tok_mask[range(QL)] = 2
            tok_mask[bos_idx] = 2
            self.attn_cnt += QL + len(bos_idx)
            

        # QDS-Transformer (Q)
        if self.attn_pattern == 3:
            tok_mask[range(QL)] = 2
            tok_mask[sent_locs[0]] = 2
            self.attn_cnt += QL + 1

        # QDS-Transformer (S)
        if self.attn_pattern == 4:
            tok_mask[0] = 2
            tok_mask[bos_idx] = 2
            self.attn_cnt += 1 + len(bos_idx)
        
        self.attn_cnt += self.window_size * 2
        self.attn_total += L

        # Process masked lm.
        if masked_lm:
            masked_lm_labels = np.ones([len(input_ids)], dtype=np.long) * (-100)
            tok_idx = [i for i, x in enumerate(input_ids) if x != 0]
            random.shuffle(tok_idx)
            LM_L = int(len(tok_idx) * 15 / 100)
            tok_idx = tok_idx[:LM_L]
                
            input_ids = np.array(input_ids, dtype=np.long)
            masked_lm_labels[tok_idx] = input_ids[tok_idx]
            input_ids[tok_idx] = 50264

            return input_ids, tok_mask, masked_lm_labels

        return input_ids, tok_mask, sent_locs, sent_mask


    # Model Data Utility Functions
    def create_empty_batch(self):
        return {'rel': [],
                'label': [],
                'docid': [],
                'qid': [],
                'input_ids': [],
                'tok_mask': [],
                'sent_locs': [],
                'sent_mask': []}

    def push_data_into_batch(self, rel, qid, docid, batch):
        batch['rel'].append(rel)
        batch['label'].append([1.0 if rel > 0 else 0.0]) 
        batch['docid'].append(docid)
        batch['qid'].append(qid)
        input_ids, tok_mask, sent_locs, sent_mask = self.get_inputs(qid, docid)
        batch['input_ids'].append(input_ids)
        batch['tok_mask'].append(tok_mask)
        batch['sent_locs'].append(sent_locs)
        batch['sent_mask'].append(sent_mask)


    def to_tensor(self, batch):
        return {
                'label': torch.tensor(batch['label'], dtype=torch.float),
                'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
                'docid': batch['docid'],
                'qid': batch['qid'],
                'tok_mask': torch.tensor(batch['tok_mask'], dtype=torch.float),
                'sent_locs': torch.tensor(
                    batch['sent_locs'], dtype=torch.long),
                'sent_mask': torch.tensor(
                    batch['sent_mask'], dtype=torch.float)}
    
    def eval_qid_batch_iter(self, qid, onlyrel=False):
        assert(qid in self.candidates)
        cur_batch = self.create_empty_batch()
        for docid in self.candidates[qid]:
            rel = self.qrels[(qid, docid)]
            if rel == 0 and onlyrel: continue
            self.push_data_into_batch(rel, qid, docid, cur_batch)
            if len(cur_batch['label']) >= self.batch_size:
                yield self.to_tensor(cur_batch), cur_batch['rel']
                cur_batch = self.create_empty_batch()

        if len(cur_batch['label']) > 0:
            yield self.to_tensor(cur_batch), cur_batch['rel']
        

    def batch_iter(self):
        random.shuffle(self.pos_records)

        cur_batch = self.create_empty_batch()
        for data in self.pos_records:
            # Load data into the current batch.
            qid, docid = data
            for i in range(self.num_neg_samples):
                self.push_data_into_batch(0, qid,
                        random.choice(self.neg_lookup[qid]), cur_batch)
                if len(cur_batch['label']) >= self.batch_size:
                    yield self.to_tensor(cur_batch)
                    cur_batch = self.create_empty_batch()

            self.push_data_into_batch(1, qid, docid, cur_batch)

            if len(cur_batch['label']) >= self.batch_size:
                yield self.to_tensor(cur_batch)
                cur_batch = self.create_empty_batch()

        if len(cur_batch['label']) > 0:
            yield self.to_tensor(cur_batch)

