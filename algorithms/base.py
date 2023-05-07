from __future__ import annotations
from torch_geometric.data import Batch
from torch import Tensor, from_numpy, no_grad
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

class graph_embedder(ABC):
    '''
        Abstract class for graph embedding. Used for inheriting
    various methods, that need to be inherited, as well as demands
    some to be developed!
    '''

    @abstractmethod
    def forward(self, batch: Batch, answers: Tensor, corrupted: Tensor)->tuple[Tensor,Tensor,Tensor]:
        '''return loss (batch loss), golden score, corrupted score'''
        pass

    @abstractmethod
    def score(self, query_embs: Tensor, answers: Tensor)->Tensor:
        '''return score (appropriate for the embedder)'''
        pass

    @abstractmethod
    def embed_query(self, batch: Batch)->Tensor:
        '''return embedding of graph batch'''
        pass

    @abstractmethod
    def normalize(self)->None:
        '''If needed, should decide how to normalize embeddings'''
        pass

    def evaluate(self, query: Batch, answer: Tensor, unsqueeze: bool = False):
        #evaluate score of a query and proposed answer!
        embedded_query = self.embed_query(query)
        if unsqueeze:
            '''If unsqueeze, it expands [batch, query_emb]
            to [batch, 1, query_emb], so as to anticipate 
            the change in shape of answer... used when 
            answer.size is (batch_size, n_entities)'''
            embedded_query = embedded_query.unsqueeze(1) 
        return self.score(embedded_query, answer)

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
