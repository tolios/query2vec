from __future__ import annotations
from torch_geometric.data import Batch
from torch import Tensor, from_numpy, no_grad, device, logical_not, argsort, relu, clone, stack, min, max, cat, softmax, sum, tensor, float
from torch.nn.functional import normalize
from torch.nn import Module, Embedding, ModuleList, init, Dropout, LayerNorm, ModuleList
from numpy import ndarray
from abc import ABC, abstractmethod

def dist_box(b: Tensor, x: Tensor, alpha: Tensor, box_sep: int)->Tensor:
    #* b is the box embedding, x is the point, alpha is a parameter
    if len(b.shape) == 2:
        centre, off = b[:, :box_sep], b[:, box_sep:]
    else:
        centre, off = b[:,:, :box_sep], b[:,:, box_sep:] #used when we unsqueeze
    q_max = centre + off
    q_min = centre - off
    # calculate dist_out, dist_in
    dist_out = (relu(x - q_max) + relu(q_min - x)).norm(dim=-1, p=1)
    dist_in = (centre - min(q_max, max(q_min, x))).norm(dim=-1, p=1)

    return dist_out + (alpha*dist_in)

def box_embs(x: Tensor, box_sep: int)->Tensor:
    # box embeddings are a centre vector and and off POSITIVE vector!
    # first seperate!
    centre, off = x[:, :box_sep], x[:, box_sep:]
    # make off positive (they act as margins)
    off = relu(off)
    # return back together
    return cat((centre, off), 1)

def root_embs(x: Tensor, edge_index: Tensor, batch_id: Tensor)->Tensor:
    '''
        This function receives the embeddings, edge_indexes and batch_ids of
        a given Batch graph, returns the root embeddings!

        root is simply the node of the query graph that has no outward links
        acting as the placeholder variable for the answers!
    '''
    heads, tails = edge_index[0], edge_index[1]
    # if in tails but not in heads, it is root!
    is_root = logical_not((tails.unsqueeze(1) == heads).any(1))
    root_pos = tails[is_root]
    root_batch_id = batch_id[root_pos]
    # given the batch id sort root_pos according to it...
    root_pos = root_pos[argsort(root_batch_id)]
    #make unique pos to avoid registering the same root many times 2i, 3i, pi problem!
    root_pos = root_pos.unique_consecutive()

    return x[root_pos]

class conv_pipe(Module):
    '''
        This module neatly packs all convolution layers in a place
    and forwards all layer embeddings!
    '''
    def __init__(self, conv_layers: ModuleList, dynamic=True, p = 0.2):
        super().__init__()
        self.conv_layers = conv_layers
        self.g_norms = ModuleList([LayerNorm(l.out_channels) for l in conv_layers])
        self.dropouts = ModuleList([Dropout(p = p) for _ in self.g_norms])
        self.dynamic = dynamic #if falsoe simply acts like typical iteration (no clones) 

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor)->Tensor:
        emb_layers = []
        for dropout, g_norm, layer in zip(self.dropouts, self.g_norms, self.conv_layers):
            x = layer(x, edge_index=edge_index, edge_type=edge_attr.squeeze(1))
            x = g_norm(x)
            x = dropout(x)
            x = relu(x)
            # we have to clone to keep the diff structure for backprop
            if self.dynamic:
                emb_layers.append(clone(x))
        #! DYNAMIC EMBEDDING
        #! NEEDS layers to have same emb dim!
        # simply acting with [-1] we get the last x
        if not self.dynamic:
            emb_layers.append(x) #only keeping last...
        return stack(emb_layers)

class graph_embedding(Module):
    def __init__(self, num_entities, emb_dim):
        super().__init__()                              
        self.entity_emb = Embedding(num_entities+1, emb_dim, padding_idx=0) #entities have a padding of zeros!(maybe no)
        init.xavier_uniform_(self.entity_emb.weight)
    def forward(self, g_batch):
        #*  Receives a batch of graphs, containing entity/relationship ids,
        #* returns embedded batch of graphs ...
        x = self.entity_emb(g_batch.x).squeeze(-2)
        #construct data object...
        return Batch(x=x, edge_index=g_batch.edge_index,
                edge_attr=g_batch.edge_attr, 
                batch=g_batch.batch,
                depth=g_batch.depth, #include for dynamic emb
                ptr=g_batch.ptr) 
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

    def __init__(self, num_entities, emb_dim, T_emb = 0.25):
        super().__init__()
        #Model weights!
        self.graph_embedding = graph_embedding(num_entities, emb_dim)
        self.device = device("cpu") #puts weights to cpu
        self.register_buffer('T_emb', tensor([T_emb], dtype=float))

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
        cs = self.score(batch, corrupted, unsqueeze=True)
        corrupted_score = sum(cs*softmax(cs/(self.T_emb), dim=-1), dim=-1) # avg corr score! #FIXME - this is highly controversial
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
