import sys
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
import numpy as np
import pickle

from tqdm import tqdm

import utils
import modellib

try:
    import ujson as json
except:
    import json

MODEL_DICT = {
        'QDST': modellib.QDSTModel,
}

class Experiment:
    def __init__(self, FLAGS, cfg, dumpflag=True):
        # GPU setup.
        self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        print('device: {}'.format(self.device), file=sys.stderr)
        self.pred_file = FLAGS.pred_file

        # Declare a model.
        self.prev_mrr = None
        self.prev_model = None
        self.model = MODEL_DICT[FLAGS.model](FLAGS)
        if FLAGS.load_model != '' and os.path.exists(FLAGS.load_model):
            print('Load previous model from {}'.format(
                FLAGS.load_model), file=sys.stderr)
            self.model.load_state_dict(torch.load(FLAGS.load_model))

        self.model.to(self.device)

        # Build experimental settings.
        self.batch_size = FLAGS.batch_size
        self.num_neg_samples = FLAGS.num_neg_samples
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
                reduction='mean',
                pos_weight=torch.FloatTensor(
                    [FLAGS.num_neg_samples]).to(self.device))
        self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=FLAGS.learning_rate)
        self.PATH_MODEL_PREFIX = '{}/{}'.format(cfg['path_model'], FLAGS.model_prefix)
        self.best_eval_mrr = -1e+10

        # Dump flags
        if dumpflag:
            with open('{}/model.{}.flags'.format(
                self.PATH_MODEL_PREFIX, self.model.get_name()), 'w') as wp:
                print(json.dumps(FLAGS.flag_values_dict()), file=wp)

        # Build tensorboard settings.
        self.tb = FLAGS.tb
        self.n_iter = 0
        if self.tb:
            TB_RUNS_DIR = cfg['path_tb']
            TB_SUFFIX = '{}_{}_{}'.format(
                    FLAGS.model, FLAGS.model_prefix, time.time())
            print('Tensorboard Dir: {}'.format(TB_RUNS_DIR), file=sys.stderr)
            print('Tensorboard Suffix: {}'.format(TB_SUFFIX), file=sys.stderr)
            self.tb_per_iter = FLAGS.tb_per_iter
            self.tb_writer = SummaryWriter(
                    log_dir='{}/{}'.format(TB_RUNS_DIR, TB_SUFFIX))


    def dump_model(self, PATH_MODEL_PREFIX, FILE_SUFFIX):
        file_name = '{}/model.{}.{}.pt'.format(
                PATH_MODEL_PREFIX, self.model.get_name(), FILE_SUFFIX)
        torch.save(self.model.state_dict(), file_name)

    def load_model(self, PATH_MODEL_PREFIX, FILE_SUFFIX):
        file_name = '{}/model.{}.{}.pt'.format(
                PATH_MODEL_PREFIX, self.model.get_name(), FILE_SUFFIX)
        self.model.load_state_dict(torch.load(file_name))


    def profile_train(self,
            train_data,
            eval_data=None,
            test_data=None,
            num_sample_eval=None,
            silent=False):
        num_batches = len(train_data.pos_records) * (self.num_neg_samples + 1)
        num_batches = num_batches + self.batch_size - 1
        num_batches = num_batches // self.batch_size
        self.model.train()

        total_time = 0.0
        iter_cnt = 0
        for data in tqdm(
                train_data.batch_iter(),
                desc='Train', total=num_batches, ncols=80, disable=silent):
            start_time = time.time()
            try:
                y_pred = self.model(
                        data['input_ids'].to(self.device),
                        data['tok_mask'].to(self.device),
                        data['sent_locs'].to(self.device),
                        data['sent_mask'].to(self.device))
            except:
                print(data)
                y_pred = self.model(
                        data['input_ids'].to(self.device),
                        data['tok_mask'].to(self.device),
                        data['sent_locs'].to(self.device),
                        data['sent_mask'].to(self.device))
                sys.exit(0)

            loss = self.loss_fn(y_pred, data['label'].to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_time += time.time() - start_time
            iter_cnt += 1
            if iter_cnt == 100: break

        return total_time / 100.0


    def profile_eval(self, eval_data, num_sample_eval, desc='', silent=False):
        self.model.eval()


        total_time = 0.0
        iter_cnt = 0
        with torch.no_grad():
            for qid in tqdm(eval_data.get_qid_list(num_sample_eval),
                    desc=desc, ncols=80, disable=silent):
                for data, y_true in eval_data.eval_qid_batch_iter(qid):
                    start_time = time.time()
                    y_pred = self.model(
                        data['input_ids'].to(self.device),
                        data['tok_mask'].to(self.device),
                        data['sent_locs'].to(self.device),
                        data['sent_mask'].to(self.device))
                    total_time += time.time() - start_time
                    iter_cnt += 1
                    if iter_cnt == 100: break
                if iter_cnt == 100: break

        return total_time / 100.0


    def attn_eval(self, eval_data, num_sample_eval, desc='', silent=False):
        self.model.eval()
        
        for i, layer in enumerate(self.model.base.encoder.layer):
            layer.attention.self.output_attentions = True

        output_list = []

        with torch.no_grad():
            for qid in tqdm(eval_data.get_qid_list(num_sample_eval),
                    desc=desc, ncols=80, disable=silent):
                for data, y_true in eval_data.eval_qid_batch_iter(qid, onlyrel=True):
                    if y_true == 0: continue

                    attns = self.model.attn_call(
                        data['input_ids'].to(self.device),
                        data['tok_mask'].to(self.device),
                        data['sent_locs'].to(self.device),
                        data['sent_mask'].to(self.device))

                    attns = [x.cpu().detach().numpy()[0] for x in attns]
                    attns = np.stack(attns, axis=0)
                    input_ids = data['input_ids'].numpy()[0]
                    tokens = eval_data.tokenizer.convert_ids_to_tokens(input_ids)
                    tokens = ['[SEP]' if x == '<s>' else x for x in tokens]
                    tokens[0] = '[CLS]'

                    output_dict = {
                            'query': eval_data.query[data['qid'][0]],
                            'tokens': tokens,
                            'tok_mask': data['tok_mask'].numpy()[0],
                            'attns': attns,
                            'input_ids': input_ids
                    }
                    output_list.append(output_dict)

        with open(self.pred_file, 'wb') as wp:
            pickle.dump(output_list, wp)



    def train(self,
            train_data,
            eval_data=None,
            test_data=None,
            num_sample_eval=None,
            silent=False):
        num_batches = len(train_data.pos_records) * (self.num_neg_samples + 1)
        num_batches = num_batches + self.batch_size - 1
        num_batches = num_batches // self.batch_size
        self.model.train()
        train_losses = []
        for data in tqdm(
                train_data.batch_iter(),
                desc='Train', total=num_batches, ncols=80, disable=silent):
            try:
                y_pred = self.model(
                        data['input_ids'].to(self.device),
                        data['tok_mask'].to(self.device),
                        data['sent_locs'].to(self.device),
                        data['sent_mask'].to(self.device))
            except:
                print(data)
                y_pred = self.model(
                        data['input_ids'].to(self.device),
                        data['tok_mask'].to(self.device),
                        data['sent_locs'].to(self.device),
                        data['sent_mask'].to(self.device))
                sys.exit(0)

            loss = self.loss_fn(y_pred, data['label'].to(self.device))
            train_losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Dump tensorboard results.
            if self.tb:
                self.tb_writer.add_scalar('Loss/Train', loss.item(), self.n_iter)
                if self.n_iter > 0 and self.n_iter % self.tb_per_iter == 0:
                    if eval_data == None or num_sample_eval == None: continue
                    eval_results = self.eval(
                            eval_data, num_sample_eval, 'Midpoint Evaluation (Dev)')
                    self.tb_writer.add_scalar('Eval_cut10/NDCG', eval_results['ndcg10'], self.n_iter)
                    self.tb_writer.add_scalar('Eval_cut10/MRR', eval_results['mrr10'], self.n_iter)
                    self.tb_writer.add_scalar('Eval_cut10/MAP', eval_results['map10'], self.n_iter)
                    self.tb_writer.add_scalar('Eval_overall/NDCG', eval_results['ndcg'], self.n_iter)
                    self.tb_writer.add_scalar('Eval_overall/MRR', eval_results['mrr'], self.n_iter)
                    self.tb_writer.add_scalar('Eval_overall/MAP', eval_results['map'], self.n_iter)

                    if eval_results['mrr10'] > self.best_eval_mrr:
                        self.tb_writer.add_scalar('Eval_cut10/BestMRR', eval_results['mrr10'], self.n_iter)
                        self.best_eval_mrr = eval_results['mrr10']
                        self.dump_model(self.PATH_MODEL_PREFIX, 'best.{}'.format(self.n_iter))
                        if test_data == None or num_sample_eval == None: continue
                        self.pred_file = '{}/pred.{}'.format(self.PATH_MODEL_PREFIX, self.n_iter)
                        eval_results = self.eval_dump(
                                test_data, num_sample_eval, 'Midpoint Evaluation (Test)')
                        self.tb_writer.add_scalar('Test_cut10/NDCG', eval_results['ndcg10'], self.n_iter)
                        self.tb_writer.add_scalar('Test_cut10/MRR', eval_results['mrr10'], self.n_iter)
                        self.tb_writer.add_scalar('Test_cut10/MAP', eval_results['map10'], self.n_iter)
                        self.tb_writer.add_scalar('Test_overall/NDCG', eval_results['ndcg'], self.n_iter)
                        self.tb_writer.add_scalar('Test_overall/MRR', eval_results['mrr'], self.n_iter)
                        self.tb_writer.add_scalar('Test_overall/MAP', eval_results['map'], self.n_iter)

            self.n_iter += 1

        return np.mean(train_losses)


    def eval(self, eval_data, num_sample_eval, desc='', silent=False):
        self.model.eval()
        evaler = utils.Evaluater()
        with torch.no_grad():
            qid_cnt = 0
            for qid in tqdm(eval_data.get_qid_list(num_sample_eval),
                    desc=desc, ncols=80, disable=silent):
                y_true_all = []
                y_pred_all = []
                checked = set()
                for data, y_true in eval_data.eval_qid_batch_iter(qid):
                    y_pred = self.model(
                        data['input_ids'].to(self.device),
                        data['tok_mask'].to(self.device),
                        data['sent_locs'].to(self.device),
                        data['sent_mask'].to(self.device))
                    y_pred = y_pred.cpu().detach().numpy()
                    y_true_all.extend(y_true)
                    y_pred_all.extend([x[0] for x in y_pred])
                    for i in range(len(data['docid'])):
                        d = data['docid'][i]
                        checked.add(d)
                for d, r in eval_data.qrels_by_q[qid]:
                    if d not in checked:
                        y_pred_all.append(-1e+100)
                        y_true_all.append(int(r))

                evaler.eval(y_pred_all, y_true_all)
                qid_cnt += 1
                if qid_cnt % 10 == 0:
                    print(evaler.summary())
                    with open('xxxx', 'w') as wp:
                        print(evaler.summary(), file=wp)

        return evaler.summary()


    def eval_dump(self, eval_data, num_sample_eval, desc='', silent=False):
        self.model.eval()
        evaler = utils.Evaluater()
        feature_dict = {}
        hidden_feature_dict = {}
        with open(self.pred_file, 'w') as wp:
            with torch.no_grad():
                for qid in tqdm(eval_data.get_qid_list(num_sample_eval),
                        desc=desc, ncols=80, disable=silent):
                    y_true_all = []
                    y_pred_all = []
                    docid_all = []
                    features_all = []
                    hidden_features_all = []
                    checked = set()
                    for data, y_true in eval_data.eval_qid_batch_iter(qid):
                        y_pred = self.model(
                            data['input_ids'].to(self.device),
                            data['tok_mask'].to(self.device),
                            data['sent_locs'].to(self.device),
                            data['sent_mask'].to(self.device))

                        y_pred = y_pred.cpu().detach().numpy()
                        y_true_all.extend(y_true)
                        y_pred_all.extend([x[0] for x in y_pred])
                        docid_all.extend(data['docid'])
                        features_all.append(self.model.final_features.cpu().detach().numpy())
                        hidden_features_all.append(self.model.final_hidden_features.cpu().detach().numpy())

                        for i in range(len(data['docid'])):
                            q = qid
                            d = data['docid'][i]
                            checked.add(d)

                    for d, r in eval_data.qrels_by_q[qid]:
                        if d not in checked and r > 0:
                            y_pred_all.append(-1e+100)
                            y_true_all.append(int(r))
                            docid_all.append(d)

                    evaler.eval(y_pred_all, y_true_all)

                    assert(len(y_pred_all) == len(y_true_all) == len(docid_all))

                    hidden_features_all = np.concatenate(hidden_features_all)
                    for i in range(len(hidden_features_all)):
                        hidden_feature_dict[(qid, docid_all[i])] = hidden_features_all[i, :]


                    features_all = np.concatenate(features_all)
                    for i in range(len(features_all)):
                        feature_dict[(qid, docid_all[i])] = features_all[i, :]
                        print('{} Q0 {} {} {} XXX'.format(
                            qid, docid_all[i], i + 1, y_pred_all[i]),
                            file=wp)
                    wp.flush()

        with open(self.pred_file + '.features', 'wb') as wp:
            pickle.dump(feature_dict, wp)

        with open(self.pred_file + '.hidden_features', 'wb') as wp:
            pickle.dump(feature_dict, wp)
        return evaler.summary()

