# MOST GEBERAL MODEL possible... bar the embedding type

from __future__ import annotations
from torch_geometric.nn.conv import FastRGCNConv, RGATConv
from torch_scatter import scatter_add, scatter_max, scatter_mean
import torch
import torch.nn as nn
from .base import qa_embedder, conv_pipe, dist_box, root_embs

class Model(qa_embedder): 
    def __init__(self, num_entities, num_relationships, num_bases = None, num_blocks = None,
                kernel = "rgcn", T = 0., aggregation = "sum", dynamic = False,
                emb_dim = 50, conv_dims=[100],linear_dims=[50], heads = 1, p=0.2, margin = 1.0):
        super().__init__(num_entities, emb_dim)
        self.num_entities = num_entities
        self.num_relationships = num_relationships
        self.kwargs = {
            'kernel': kernel,
            'T': T,
            'aggregation': aggregation,
            'dynamic': dynamic,
            'emb_dim': emb_dim,
            'conv_dims': conv_dims,
            'num_bases': num_bases,
            'num_blocks' : num_blocks,
            'linear_dims': linear_dims,
            'heads': heads,
            'p': 0.2,
            'margin': margin,
        }
        if T > 0:
            self.Tzero = False
        elif T == 0:
            self.Tzero = True
        else:
            raise
        self.kernel = kernel
        if self.kernel == "rgcn":
            kernel_ = FastRGCNConv
        elif self.kernel == "rgat":
            kernel_ = RGATConv
        else:
            raise
        self.dynamic = dynamic
        self.aggregation = aggregation

        # nn.ModuleList allows to the lists to be tracked by .to for gpus!
        self.conv_pipe = conv_pipe(nn.ModuleList([kernel_(l, r, num_relationships, 
                num_bases=num_bases, num_blocks=num_blocks, is_sorted=True) \
                    for l, r in zip([emb_dim]+conv_dims, ([[]]+conv_dims)[1:])]), dynamic=dynamic)
        self.linear_layers = nn.ModuleList([nn.Linear(l, r) for l, r in zip([conv_dims[-1]]+linear_dims+[[]], ([[]]+linear_dims+[heads*emb_dim])[1:])])
        self.dropouts = nn.ModuleList([nn.Dropout(p=p) for _ in linear_dims])
        self.reshaper = lambda x: x.reshape(-1, heads, emb_dim) # from (-1, heads*emb_dim) to (-1, heads, emb_dim)
        self.register_buffer('T', torch.tensor([T], dtype=torch.float)) #temperature!
        self.register_buffer('margin', torch.tensor([margin], dtype=torch.float))
    
    def loss(self, golden_score, corrupted_score):
        #REVIEW - multiply all with self.T and change thesis definition
        if not self.Tzero:
            return -(golden_score - corrupted_score - self.margin) + self.T*torch.log((torch.exp((golden_score - corrupted_score - self.margin)/(self.T)) + 1)/2) #NOTE - div by 2, inside log so we remove log 2 as min
        else:
            return torch.relu(-(golden_score - corrupted_score - self.margin))

    def _score(self, query_embs, answers):
        answers = answers.unsqueeze(-2)
        norm = torch.norm(query_embs, p = 2, dim = -1)*torch.norm(answers, p = 2, dim = -1) # type: ignore
        scores = torch.sum(query_embs*answers, dim=-1)/norm
        #get max of those scores...
        return torch.max(scores, -1)[0]

    #FIXME - check memory problems
    def _embed_query(self, batch):
        x, edge_index, edge_attr, batch_id = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = self.conv_pipe(x, edge_index, edge_attr)
        # most serious step
        if self.dynamic:
            depth = batch.depth - 1 #we use depth for dynamic embedding! (-1 for correct index pos)
            x = x[depth[batch_id]] # shape [nodes, nodes, emb] all node embs per node for correct depth
            nodes = x.shape[0]
            x = x[torch.arange(0, nodes), torch.arange(0, nodes)] # select appropriate depth emb shape [nodes, emb]
        else:
            x = x[-1] #simply get the last for all
        for layer, dropout in zip(self.linear_layers[:-1], self.dropouts):
            x = layer(x)
            x = torch.relu(x)
            x = dropout(x)
        #last layer ...
        x = self.linear_layers[-1](x)
        #* return aggregated embedding nodes by batch id...
        x = self.aggregate(x, edge_index, batch_id)
        #reshape to [-1, heads, emb]
        x = self.reshaper(x) # reshape for many head calculation
        return x

    def aggregate(self, x, edge_index, batch_id):
        #* This method receives the batch node embeddings and their corresponding batch member ids,
        #* and aggregates by grouping by the batch member id...
        if self.aggregation == "sum":
            return scatter_add(x, batch_id, dim=0)
        elif self.aggregation == "mean":
            return scatter_mean(x, batch_id, dim=0)
        elif self.aggregation == "max":
            return scatter_max(x, batch_id, dim=0)
        elif self.aggregation == "root":
            return root_embs(x, edge_index, batch_id)
        else:
            raise
