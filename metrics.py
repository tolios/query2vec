import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

def mean_rank(data: Dataset, model: torch.nn.Module, batch_size = 64, device=torch.device('cpu')):
    model.eval() #set for eval
    with torch.no_grad():
        plus = torch.tensor([1], dtype=torch.long, device=device)
        mean = 0.
        n_queries = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        comp = comp.to(device)
        for q, a in tqdm(loader, desc='Raw mean rank calculation'):
            q, a = q.to(device), a.to(device)
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.evaluate(q, comp, unsqueeze=True) #unsqueeze for shape
            a = a.view(-1, 1)
            #calculating indices for sorting...
            _, _indices = torch.sort(scores, dim = 1, descending=True) #! changed: scores goes big!
            #adding index places... +1 very important for correct positioning!!!
            mean += torch.sum(torch.eq(_indices+plus,a).nonzero()[:, 1]).item()
        return mean/(n_queries)

def hits_at_N(data: Dataset, model: torch.nn.Module, N = 10, batch_size = 64, device=torch.device('cpu')):
    model.eval() #set for eval
    with torch.no_grad():
        plus = torch.tensor([1], dtype=torch.long, device=device)
        hits = 0
        n_queries = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        #useful tensors...
        zero_tensor, one_tensor = torch.tensor([0]), torch.tensor([1])
        zero_tensor, one_tensor = zero_tensor.to(device), one_tensor.to(device)
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        comp = comp.to(device)
        for q, a in tqdm(loader, desc=f'Raw hits@{N} calculation'):
            q, a = q.to(device), a.to(device)
            #calculating head and tail energies for prediction
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.evaluate(q, comp, unsqueeze=True) #unsqueeze for shape
            a = a.view(-1, 1)
            #calculating indices for topk...
            _, _indices = torch.topk(scores, N, dim=1, largest=True) #! changed: scores goes big!
            #summing hits... +1 very important for correct positioning!!!
            hits += torch.where(torch.eq(_indices+plus, a), one_tensor, zero_tensor).sum().item()
        #return total hits over 2*n_queries!
        return hits/(n_queries)
