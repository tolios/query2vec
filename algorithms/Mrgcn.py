from __future__ import annotations
from torch_geometric.nn.conv import FastRGCNConv
from torch_scatter import scatter_add
import torch
import torch.nn as nn
from .base import graph_embedder, graph_embedding

class Model(graph_embedder, nn.Module): 
    def __init__(self, num_entities, num_relationships, num_bases = None, num_blocks = None,
            emb_dim = 50, heads=3, conv_dims=[100],linear_dims=[50], p=0.2, margin = 1.0):
        super().__init__()
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
            'heads': heads
        }
        #Model weights!
        self.graph_embedding = graph_embedding(num_entities, emb_dim)
        # nn.ModuleList allows to the lists to be tracked by .to for gpus!
        self.conv_layers = nn.ModuleList([FastRGCNConv(l, r, num_relationships, 
                        num_bases=num_bases, num_blocks=num_blocks, is_sorted=True) \
                            for l, r in zip([emb_dim]+conv_dims, ([[]]+conv_dims)[1:])])
        self.linear_layers = nn.ModuleList([nn.Linear(l, r) \
                for l, r in zip([conv_dims[-1]]+linear_dims+[[]], ([[]]+linear_dims+[heads*emb_dim])[1:])])
        self.dropouts = nn.ModuleList([nn.Dropout(p=p) for _ in linear_dims])
        #used to implement loss! reduction = none, so it is used for outputing batch losses (later we sum them)
        self.criterion = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.reshaper = lambda x: x.reshape(-1, heads, emb_dim)
        self.register_buffer('target', torch.tensor([1], dtype=torch.long))

    def forward(self, batch, answers, corrupted):
        #* Main method of the class used for training...
        batch = self.embed_query(batch) #calculate query embedding
        golden_score = self.score(batch, answers)
        corrupted_score = self.score(batch, corrupted)
        #now return loss!
        loss = self.criterion(golden_score, corrupted_score, self.target)
        return loss, golden_score, corrupted_score

    #! should not be here ?
    def score(self, query_embs, answers):
        #* Embed answer nodes, then calculate score!
        answers = self.graph_embedding.embed_entities(answers)
        answers = answers.unsqueeze(-2)
        norm = torch.norm(query_embs, p = 2, dim = -1)*torch.norm(answers, p = 2, dim = -1) # type: ignore
        scores = torch.sum(query_embs*answers, dim=-1)/norm
        #get max of those scores...
        return torch.max(scores, -1)[0]

    def embed_query(self, batch):
        batch = self.graph_embedding(batch)
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
        #aggregate
        x = self.aggregate(x, batch_id)
        #reshape to [-1, heads, emb]
        x = self.reshaper(x)

        return x

    def aggregate(self, x, batch_id):
        #* This method receives the batch node embeddings and their corresponding batch member ids,
        #* and aggregates by grouping by the batch member id...
        return scatter_add(x, batch_id, dim=0)
    
    def normalize(self):
        #with this method, all normalizations are performed.
        #To be used before mini-batch training in each epoch.
        self.graph_embedding.normalize()
