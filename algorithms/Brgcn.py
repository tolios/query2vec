from __future__ import annotations
from torch_geometric.nn.conv import FastRGCNConv
from torch_scatter import scatter_add
import torch
import torch.nn as nn
from .base import qa_embedder, box_embs, dist_box

class Model(qa_embedder): 
    def __init__(self, num_entities, num_relationships, num_bases = None, num_blocks = None,
                emb_dim = 50, conv_dims=[100],linear_dims=[50], alpha = 0.1, gamma = 0.1, p=0.2, margin = 1.0):
        super().__init__(num_entities, emb_dim)
        self.num_entities = num_entities
        self.num_relationships = num_relationships
        self.kwargs = {
            'emb_dim': emb_dim,
            'conv_dims': conv_dims,
            'num_bases': num_bases,
            'num_blocks' : num_blocks,
            'linear_dims': linear_dims,
            'p': 0.2,
            'margin': margin,
        }
        # nn.ModuleList allows to the lists to be tracked by .to for gpus!
        self.conv_layers = nn.ModuleList([FastRGCNConv(l, r, num_relationships, 
                        num_bases=num_bases, num_blocks=num_blocks, is_sorted=True) \
                            for l, r in zip([emb_dim]+conv_dims, ([[]]+conv_dims)[1:])])
        self.linear_layers = nn.ModuleList([nn.Linear(l, r) for l, r in zip([conv_dims[-1]]+linear_dims+[[]], ([[]]+linear_dims+[2*emb_dim])[1:])]) # 2*emb because box embs!
        self.dropouts = nn.ModuleList([nn.Dropout(p=p) for _ in linear_dims])
        #used to implement loss! reduction = none, so it is used for outputing batch losses (later we sum them)
        self.register_buffer('alpha', torch.tensor([alpha], dtype=torch.float))
        self.register_buffer('gamma', torch.tensor([gamma], dtype=torch.float))
        #* Purely for box embedding!
        self.box_sep = emb_dim # useful for seperation of Centre and Off of box emb.
    
    def loss(self, golden_score, corrupted_score):
        return -(torch.log(0.5*(golden_score+1) + 0.01) + torch.log(1.01 - (0.5*(corrupted_score+1)))) # log of mean vs in paper mean of log

    def _score(self, query_embs, answer_embs):
        return (2*torch.sigmoid(self.gamma - dist_box(query_embs, answer_embs, self.alpha, self.box_sep))/torch.sigmoid(self.gamma))-1 # for dist = 0, score = 1 else score < 1

    def _embed_query(self, batch):
        x, edge_index, edge_attr, batch_id = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        for layer in self.conv_layers:
            x = layer(x, edge_index=edge_index, edge_type=edge_attr.squeeze(1))
            x = torch.relu(x)
        for layer, dropout in zip(self.linear_layers[:-1], self.dropouts):
            x = layer(x)
            x = torch.relu(x)
            x = dropout(x)
        #last layer ...
        x = self.linear_layers[-1](x)
        #* aggregated embedding nodes by batch id...
        x = self.aggregate(x, batch_id)
        #* create box_embeddings!
        return box_embs(x, self.box_sep)

    def aggregate(self, x, batch_id):
        #* This method receives the batch node embeddings and their corresponding batch member ids,
        #* and aggregates by grouping by the batch member id...
        return scatter_add(x, batch_id, dim=0)
