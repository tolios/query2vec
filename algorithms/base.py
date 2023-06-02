from __future__ import annotations
from torch_geometric.data import Batch
from torch import Tensor, from_numpy, no_grad, mean, device
from torch.nn.functional import normalize
from torch.nn import Module, Embedding
from numpy import ndarray
from abc import ABC, abstractmethod

class graph_embedding(Module):
    def __init__(self, num_entities, emb_dim):
        super().__init__()                              
        self.entity_emb = Embedding(num_entities+1, emb_dim, padding_idx=0) #entities have a padding of zeros!(maybe no)
    def forward(self, g_batch):
        #*  Receives a batch of graphs, containing entity/relationship ids,
        #* returns embedded batch of graphs ...
        x = self.entity_emb(g_batch.x).squeeze(-2)
        #construct data object...
        return Batch(x=x, edge_index=g_batch.edge_index,
                edge_attr=g_batch.edge_attr, batch=g_batch.batch, ptr=g_batch.ptr) 
    def embed_entities(self, batch):
        return self.entity_emb(batch)
    def normalize(self):
        #speeds up training, by normalizing embeddings!!!
        self.entity_emb.weight.data = normalize(self.entity_emb.weight.data, dim = 1, p = 2)

class qa_embedder(Module, ABC):
    '''
        Abstract class for graph embedding. Used for inheriting
    various methods, that need to be inherited, as well as demands
    some to be developed!
    '''

    def __init__(self, num_entities, emb_dim):
        super().__init__()
        #Model weights!
        self.graph_embedding = graph_embedding(num_entities, emb_dim)
        self.device = device("cpu") #puts weights to cpu

    @abstractmethod
    def loss(self, golden_score: Tensor, corrupted_score: Tensor)->Tensor:
        '''
            This is the custom loss function to be implemented...
        '''
        pass

    @abstractmethod
    def _score(self, query_embs: Tensor, answer_embs: Tensor)->Tensor:
        '''return score (appropriate for the embedder)'''
        pass

    @abstractmethod
    def _embed_query(self, embedded_batch: Batch)->Tensor:
        '''method that describes how an embedded query graph will be
        transformed to an embedding Tensor like vector or more general!'''
        pass

    def embed_entities(self, entities: Tensor)->Tensor:
        return self.graph_embedding.embed_entities(entities)
    
    def embed_query(self, batch: Batch)->Tensor:
        '''return embedding of graph batch'''
        return self._embed_query(self.graph_embedding(batch))

    def forward(self, batch, answers, corrupted):
        #* Main method of the class used for training...
        batch = self.embed_query(batch) #calculate query embedding
        golden_score = self.score(batch, answers)
        corrupted_score = mean(self.score(batch, corrupted, unsqueeze=True), dim=-1) # avg corr score!
        return self.loss(golden_score, corrupted_score), golden_score, corrupted_score

    def evaluate(self, query: Batch, answer: Tensor, unsqueeze: bool = False):
        #evaluate score of a query and proposed answer!
        embedded_query = self.embed_query(query)
        return self.score(embedded_query, answer, unsqueeze=unsqueeze)
    
    def score(self, query_embs:Tensor, answers:Tensor, unsqueeze: bool=False)->Tensor:
        #* Embed answer nodes, then calculate score!
        if unsqueeze:
            #* Unsqueeze useful when answer is of shape (batch_size, n), instead of (batch_size)
            query_embs = query_embs.unsqueeze(1)
        return self._score(query_embs, self.embed_entities(answers))
    
    def normalize(self):
        #with this method, all normalizations are performed.
        #To be used before mini-batch training in each epoch.
        self.graph_embedding.normalize()

    def to(self, model_device: device):
        self.device = model_device
        super().to(model_device)

    #! SPARQL TO DAG ???
    #! NOTE THAT IT IS ONLY FOR ONE DATASET!
    #! ANSWERS TO DAG QUERY ARE EITHER THOSES THAT RESPECT THE DAG OR HAVE HIGH SCORE
    #! Predict could return rank ordered the answers! (simply produce all posible answers, score and rank them!) return answers and scores
    def predict(self,x:ndarray,edge_index:ndarray,edge_attr:ndarray,batch:ndarray,ptr:ndarray)->ndarray:
        '''
            This method uses the embed_query method to ... embed a query, after constructing it.

            Tensors needed are: x, edge_index, edge_attr, batch, ptr 

            Those, can construct a Batch graph, so as to represent a batch of queries...
        '''
        self.eval() #set to evaluation
        with no_grad():
            query = Batch(
                        x=from_numpy(x), 
                        edge_index=from_numpy(edge_index), 
                        edge_attr=from_numpy(edge_attr),
                        batch=from_numpy(batch),
                        ptr=from_numpy(ptr))
            #calc embedding!
            embedding = self.embed_query(query).numpy()
        return embedding
