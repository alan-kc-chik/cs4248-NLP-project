import gc

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer
import matplotlib
matplotlib.use('Agg')
from transformers import AutoTokenizer, BertModel


class Classify(torch.nn.Module):
    def __init__(self, params, ntags):
        super(Classify, self).__init__()
        self.params = params
        self.tokenizer = AutoTokenizer.from_pretrained("./huggingface/hub/models--google--bert_uncased_L-2_H-128_A-2")
        self.bert_model = BertModel.from_pretrained("./huggingface/hub/models--google--bert_uncased_L-2_H-128_A-2")
        self.text_encoder = LstmEncoder(params.hidden_dim, params.emb_dim)
        self.dropout = nn.Dropout(params.dropout)
        self.gcn1 = GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, params.dropout, 0.2)
        self.attentions = [GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, dropout=params.dropout,
                                               alpha=0.2, concat=True) for _ in range(0)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(params.hidden_dim, params.node_emb_dim, dropout=params.dropout,
                                           alpha=0.2, concat=False)
        self.linear_transform = nn.Linear(in_features=params.node_emb_dim,
                                          out_features=ntags)

        for name, param in self.bert_model.named_parameters():
            param.requires_grad = False


    def forward(self, input_sents):
        # if (type(input_sents) is not list) or (len(input_sents) == 0):
        #     print(f'!!! input_sents = {input_sents}')
        #     print(f'!!! type(input_sents) = {type(input_sents)}')
        # gc.collect()
        torch.cuda.empty_cache()

        if torch.cuda.is_available():
            inputs = self.tokenizer(input_sents, padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        else:
            inputs = self.tokenizer(input_sents, padding=True, truncation=True, max_length=512, return_tensors='pt')
        torch.cuda.empty_cache()
        outputs = self.bert_model(**inputs)
        torch.cuda.empty_cache()
        embeds = outputs.last_hidden_state
        # embeds = self.word_embeddings(input_sents)  # bs (sents of a doc) * max_seq_len (max sents/seq length) * emb
        h = self.text_encoder(embeds)  # bs * 100 * hidden
        h = self.dropout(F.relu(h))  # Relu activation and dropout

        # Currently it's a dummy matrix with all edge weights one
        adj_matrix = np.ones((h.size(0), h.size(0)))
        # Setting link between same sentences to 0
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = self.to_tensor(adj_matrix)

        h = F.dropout(h, self.params.dropout, training=self.training)
        h, attn = self.out_att(h, adj_matrix)
        h = F.elu(h)

        # Simple max pool on all node representations
        h, _ = h.max(dim=0)
        h = self.linear_transform(h)  # bs * ntags
        # gc.collect()
        torch.cuda.empty_cache()

        return h
    
    @staticmethod
    def to_tensor(arr):
        # list -> Tensor (on GPU if possible)
        if torch.cuda.is_available():
            tensor = torch.tensor(arr).type(torch.cuda.FloatTensor)
        else:
            tensor = torch.tensor(arr).type(torch.FloatTensor)
        return tensor


class LstmEncoder(torch.nn.Module):
    def __init__(self, hidden_dimension, embedding_dimension):
        super(LstmEncoder, self).__init__()
        self.hidden_dim = hidden_dimension
        self.lstm = nn.LSTM(embedding_dimension, hidden_dimension, batch_first=True)

    def forward(self, embeds):
        # gc.collect()
        torch.cuda.empty_cache()

        # By default a LSTM requires the batch_size as the second dimension
        # You could also use batch_first=True while declaring the LSTM module, then this permute won't be required
        # embeds = embeds.permute(1, 0, 2)  # seq_len * batch_size * embedding_dim

        # packed_input = pack_padded_sequence(embeds, seq_lens)
        # _, (hn, cn) = self.lstm(embeds)
        _, (hn, _) = self.lstm(embeds)
        # two outputs are returned. _ stores all the hidden representation at each time_step
        # (hn, cn) is just for convenience, and is hidden representation and context after the last time_step
        # _ : will be of PackedSequence type, once unpacked, you will get a tensor of size: seq_len x bs x hidden_dim
        # hn : 1 x bs x hidden_dim
        # gc.collect()
        torch.cuda.empty_cache()

        return hn[-1]  # bs * hidden_dim

