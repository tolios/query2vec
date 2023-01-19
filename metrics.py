import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def mean_rank(data: Dataset, model: torch.nn.Module, batch_size = 64):
    with torch.no_grad():
        mean = 0.
        n_queries = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        for q, a in tqdm(loader, desc='Raw mean rank calculation'):
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.predict(q, comp, unsqueeze=True) #unsqueeze for shape
            a = a.view(-1, 1)
            #calculating indices for sorting...
            _, _indices = torch.sort(scores, dim = 1)
            #adding index places...
            mean += torch.sum(torch.eq(_indices,a).nonzero()[:, 1]).item()
        return mean/(n_queries)

def hits_at_N(data: Dataset, model: torch.nn.Module, N = 10, batch_size = 64):
    with torch.no_grad():
        hits = 0
        n_queries = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        #useful tensors...
        zero_tensor = torch.tensor([0])
        one_tensor = torch.tensor([1])
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        for q, a in tqdm(loader, desc=f'Raw hits@{N} calculation'):
            #calculating head and tail energies for prediction
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.predict(q, comp, unsqueeze=True) #unsqueeze for shape
            a = a.view(-1, 1)
            #calculating indices for topk...
            _, _indices = torch.topk(scores, N, dim=1, largest=False)
            #summing hits...
            hits += torch.where(torch.eq(_indices, a), one_tensor, zero_tensor).sum().item()
        #return total hits over 2*n_queries!
        return hits/(n_queries)