from absl import app, flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy

import transformers
from longformer.longformer import Longformer, LongformerConfig

import utils

class QDSTModel(torch.nn.Module):
    def __init__(self, FLAGS):
        super(QDSTModel, self).__init__()
        self.required_graph = False
        self.construct_components()
    
    def get_name(self):
        return 'QDST'

    def construct_components(self):
        # Base model.
        self.base = Longformer.from_pretrained('longformer-base-4096/')
        self.hidden_layer = nn.Linear(
                self.base.config.hidden_size,
                self.base.config.hidden_size, bias=True)
        self.final_layer = nn.Linear(self.base.config.hidden_size, 1, bias=True)
        self.dropout = nn.Dropout(self.base.config.hidden_dropout_prob)

    def forward(self, input_ids, tok_mask, sent_locs, sent_mask):
        '''
            input_ids: torch.long (batch_size, seq_len)
            tok_mask: torch.float (batch_size, seq_len)
            sent_locs: torch.long (batch_size, sent_num)
            sent_mask: torch.float (batch_size, sent_num, 1)
        '''
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        sent_num = sent_locs.shape[1]
        # Get token embeddings (batch_size, seq_len, emb_dim).
        emb = self.base(input_ids, attention_mask=tok_mask)[1]

        self.final_features = emb
        # Output Head
        emb = self.dropout(emb)
        hidden = torch.tanh(self.hidden_layer(emb))
        self.final_hidden_features = emb

        hidden = self.dropout(hidden)
        logit = self.final_layer(hidden)
        return logit

