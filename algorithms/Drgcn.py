from __future__ import annotations
from torch_geometric.nn.conv import FastRGCNConv
from torch_scatter import scatter_add
import torch
import torch.nn as nn
from .base import qa_embedder, conv_pipe

class Model(qa_embedder): 
    def __init__(self, num_entities, num_relationships, num_bases = None, num_blocks = None,
                emb_dim = 50, conv_dims=[100],linear_dims=[50], p=0.2, margin = 1.0):
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
        self.conv_pipe = conv_pipe(nn.ModuleList([FastRGCNConv(l, r, num_relationships, 
                        num_bases=num_bases, num_blocks=num_blocks, is_sorted=True) \
                            for l, r in zip([emb_dim]+conv_dims, ([[]]+conv_dims)[1:])]))
        self.linear_layers = nn.ModuleList([nn.Linear(l, r) for l, r in zip([conv_dims[-1]]+linear_dims+[[]], ([[]]+linear_dims+[emb_dim])[1:])])
        self.dropouts = nn.ModuleList([nn.Dropout(p=p) for _ in linear_dims])
        #used to implement loss! reduction = none, so it is used for outputing batch losses (later we sum them)
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.register_buffer('target', torch.tensor([1], dtype=torch.long))
    
    def loss(self, golden_score, corrupted_score):
        return self.criterion(golden_score, corrupted_score, self.target)

    def _score(self, query_embs, answer_embs):
        norm = torch.norm(query_embs, p = 2, dim = -1)*torch.norm(answer_embs, p = 2, dim = -1) # type: ignore
        return torch.sum(query_embs*answer_embs, dim=-1)/norm
    #FIXME - check memory problems
    def _embed_query(self, batch):
        x, edge_index, edge_attr, batch_id = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        depth = batch.depth - 1 #we use depth for dynamic embedding! (-1 for correct index pos)
        x = self.conv_pipe(x, edge_index, edge_attr)
        # most serious step
        x = x[depth[batch_id]] # shape [nodes, nodes, emb] all node embs per node for correct depth
        nodes = x.shape[0]
        x = x[torch.arange(0, nodes), torch.arange(0, nodes)] # select appropriate depth emb shape [nodes, emb]
        for layer, dropout in zip(self.linear_layers[:-1], self.dropouts):
            x = layer(x)
            x = torch.relu(x)
            x = dropout(x)
        #last layer ...
        x = self.linear_layers[-1](x)
        #* return aggregated embedding nodes by batch id...
        return self.aggregate(x, batch_id) #! maybe first aggregate then MLP

    def aggregate(self, x, batch_id):
        #* This method receives the batch node embeddings and their corresponding batch member ids,
        #* and aggregates by grouping by the batch member id...
        return scatter_add(x, batch_id, dim=0)
