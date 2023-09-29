import torch
from torch_geometric.data import Dataset, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pickle
import random
import ast
from form import hashQuery

class Filter:
    '''
    This class receives train, val and test qa data so as to create a filter function!
    '''
    def __init__(self, train, val, test, 
            n_entities: int, big: int = 10e5, load_path = False):
        if not load_path:
            self.n_entities = n_entities
            self.stable_dict = self._create_stable_dict(train, val)
            self.test_dict = self._create_test_dict(test)
            self.big = big
        else:
            # train, val should be None
            self.n_entities = n_entities
            self.big = big
            self.stable_dict = self.load_stable(load_path)
            self.test_dict = self._create_test_dict(test)

        self.masks = self.compile_mask(test)
    
    def negatives(self, q_hashes, num_entities, num_negs=1, start = 1):
        # works for batch of q_hashes
        ks = []
        for h in q_hashes:
            k = []
            ans = self.stable_dict[h.item()]
            while len(k) < num_negs:
                i = random.randint(start, num_entities)
                if not (i in ans):
                    k.append(i)
            ks.append(k)
        
        return torch.Tensor(ks).type(torch.int32)
    
    def test_negatives(self, q_hashes, num_entities, num_negs=1, start = 1):
        # works for batch of q_hashes
        ks = []
        for h in q_hashes:
            k = []
            ans = self.test_dict[h.item()]
            while len(k) < num_negs:
                i = random.randint(start, num_entities)
                if not (i in ans):
                    k.append(i)
            ks.append(k)
        
        return torch.Tensor(ks).type(torch.int32)

    def compile_mask(self, test):
        '''
            creates mask for all of test (or val)
        '''
        print("compiling mask...")
        masks = dict()
        for q_hash, ans in Filter._load(test):
            #extract all answers
            as_ = self.stable_dict.get(q_hash, set()) # gets set of answers of q_hash in stable, then test and joins
            as_ = as_.union(self.test_dict.get(q_hash))
            for a_ in ans:
                #remove given entity
                masks[(q_hash, a_)] = list(as_ - {a_})
            # stack for batching...
        print("Mask compiled!")
        return masks

    def mask(self, q: Batch, a: torch.Tensor)->torch.Tensor:
        '''
        Using the hashes of queries we will create a mask.
        The mask will have -big to all entities that are answers
        except the answer given by "a". All other entities plus the given
        one will have zero. The mask is to be added to the scores so as to
        artificially lower the scores, so when ranking they wont be included!
        '''
        #first we extract the hashes and the given entities
        q_hashes = q.hash.tolist()
        a = a.tolist()
        #batch mask list
        batch_mask = []
        mask = torch.arange(1, 1+self.n_entities)
        for q_hash, a_ in zip(q_hashes, a):
            #add to batch mask
            as_ = torch.Tensor(self.masks[(q_hash, a_)])
            #finds the location where each entity is located! (marks as True)
            mask_ = torch.eq(mask,as_.unsqueeze(1)).any(0)
            mask_ = -self.big*mask_
            batch_mask.append(mask_)
        # stack for batching...
        return torch.stack(batch_mask, dim=0)
    
    def change_test(self, test):
        # this method is used when consecutive test runs happen!
        #changes test dict!
        self.test_dict = self._create_test_dict(test)
        self.masks = self.compile_mask(test)

    @classmethod
    def load_stable(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
        
    @staticmethod
    def _create_train_dict(train)->dict:
        #this function creates a dictionary which uses the query hash
        #and contains the set of corresponding answers that exist for train, val
        dict_ = dict()
        for h, ans in Filter._load(train):
            dict_[h] = set(ans)
        return dict_

    @staticmethod
    def _create_stable_dict(train, val)->dict:
        #this function creates a dictionary which uses the query hash
        #and contains the set of corresponding answers that exist for train, val
        dict_ = dict()
        for h, ans in Filter._load(train):
            dict_[h] = set(ans)

        for h, ans in Filter._load(val):
            if h not in dict_:
                dict_[h] = set(ans)
            else:
                dict_[h] = dict_[h] | set(ans)
        return dict_

    @staticmethod
    def _create_test_dict(test)->dict:
        #this function creates a dictionary for test only
        # seperable so it cna be changed!
        dict_ = dict()
        for h, ans in Filter._load(test):
            dict_[h] = set(ans)
        return dict_

    @staticmethod
    def _load(path):
        # yield
        with open(path, 'r') as f:
            for line in f:
                q, ans = ast.literal_eval(line)
                h = hashQuery(q)
                yield (h, ans)

def mean_rank(data: Dataset, model: torch.nn.Module, batch_size = 64, filter: Filter = None, device=torch.device('cpu')):
    model.eval() #set for eval
    with torch.no_grad():
        plus = torch.tensor([1], dtype=torch.long, device=device)
        mean = 0.
        n_queries = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        comp = comp.to(device)
        for q, a in tqdm(loader, desc=f'{"Filtered" if filter else "Raw"} mean rank calculation'):
            q, a = q.to(device), a.to(device)
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.evaluate(q, comp, unsqueeze=True) #unsqueeze for shape
            #applying filter if given
            if filter:
                mask = filter.mask(q, a)
                mask = mask.to(device)
                scores = scores + mask
            a = a.view(-1, 1)
            #calculating indices for sorting...
            _, _indices = torch.sort(scores, dim = 1, descending=True) #! changed: scores goes big!
            #adding index places... +1 very important for correct positioning!!!
            mean += torch.sum(1+torch.eq(_indices+plus,a).nonzero()[:, 1]).item()
        return mean/(n_queries)

def hits_at_N(data: Dataset, model: torch.nn.Module, N = 10, batch_size = 64, filter: Filter = None, device=torch.device('cpu'), disable=False):
    model.eval() #set for eval
    with torch.no_grad():
        plus = torch.tensor([1], dtype=torch.long, device=device)
        hits = 0
        hitsL, hitsR = 0, 0
        n_queries = len(data)
        nL, nR = 0, 0
        loader = DataLoader(data, batch_size = batch_size)
        #useful tensors...
        zero_tensor, one_tensor = torch.tensor([0]), torch.tensor([1])
        zero_tensor, one_tensor = zero_tensor.to(device), one_tensor.to(device)
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        comp = comp.to(device)
        for q, a in tqdm(loader, desc=f'{"Filtered" if filter else "Raw"} hits@{N} calculation', disable=disable):
            q, a = q.to(device), a.to(device)
            #calculating head and tail energies for prediction
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.evaluate(q, comp, unsqueeze=True) #unsqueeze for shape
            #applying filter if given
            if filter:
                mask = filter.mask(q, a)
                mask = mask.to(device)
                scores = scores + mask
            a = a.view(-1, 1)
            #calculating indices for topk...
            _, _indices = torch.topk(scores, N, dim=1, largest=True) #! changed: scores goes big!
            #summing hits... +1 very important for correct positioning!!!
            hits_ = torch.where(torch.eq(_indices+plus, a), one_tensor, zero_tensor).sum(-1)
            hitsL_ = ((q.edge_attr.squeeze(1)%2 == 0)*hits_).sum().item()
            hitsR_ = ((q.edge_attr.squeeze(1)%2 != 0)*hits_).sum().item()
            hits += hitsL_ + hitsR_
            hitsR += hitsR_
            hitsL += hitsL_
            nL += (q.edge_attr.squeeze(1)%2 == 0).sum().item()
            nR += (q.edge_attr.squeeze(1)%2 != 0).sum().item()

        # return total hits over n_queries!
        # print(hits/(n_queries), hitsL/nL, hitsR/nR)
        return hits/(n_queries)

def hits_at_N_Grouped(data: Dataset, model: torch.nn.Module, N = 10, batch_size = 64, filter: Filter = None, device=torch.device('cpu'), disable=False):
    model.eval() #set for eval
    with torch.no_grad():
        plus = torch.tensor([1], dtype=torch.long, device=device)
        loader = DataLoader(data, batch_size = batch_size)
        #useful tensors...
        zero_tensor, one_tensor = torch.tensor([0]), torch.tensor([1])
        zero_tensor, one_tensor = zero_tensor.to(device), one_tensor.to(device)
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        comp = comp.to(device)
        collect = dict() # collects all hits per hash
        for q, a in tqdm(loader, desc=f'{"Filtered" if filter else "Raw"} hits@{N} grouped calculation', disable=disable):
            q, a = q.to(device), a.to(device)
            #calculating head and tail energies for prediction
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.evaluate(q, comp, unsqueeze=True) #unsqueeze for shape
            #applying filter if given
            if filter:
                mask = filter.mask(q, a)
                mask = mask.to(device)
                scores = scores + mask
            a = a.view(-1, 1)
            #calculating indices for topk...
            _, _indices = torch.topk(scores, N, dim=1, largest=True) #! changed: scores goes big!
            #summing hits... +1 very important for correct positioning!!!
            hits = torch.where(torch.eq(_indices+plus, a), one_tensor, zero_tensor).sum(-1) #add result will be either 1, or 0
            for hit, hash in zip(hits, q.hash):
                hash = hash.item()
                if hash in collect:
                    collect[hash].append(hit.item())
                else:
                    collect[hash] = [hit.item()]
        #return total hits over n_queries!
        return sum([sum(collect[hash])/len(collect[hash]) if len(collect[hash])!= 0 else 0 for hash in collect])/(len(collect))

def mean_reciprocal_rank(data: Dataset, model: torch.nn.Module, batch_size = 64, filter: Filter = None, device=torch.device('cpu')):
    model.eval() #set for eval
    with torch.no_grad():
        plus = torch.tensor([1], dtype=torch.long, device=device)
        mean = 0.
        n_queries = len(data)
        loader = DataLoader(data, batch_size = batch_size)
        #creating a tensor for comparisons over all entities per batch...
        comp = torch.arange(1, model.num_entities + 1).expand(batch_size, -1)
        comp = comp.to(device)
        for q, a in tqdm(loader, desc=f'{"Filtered" if filter else "Raw"} mean reciprocal rank calculation'):
            q, a = q.to(device), a.to(device)
            if a.shape[0] != batch_size:
                #last batch...
                comp = comp[:a.shape[0]]
            #calculating scores...
            scores = model.evaluate(q, comp, unsqueeze=True) #unsqueeze for shape
            #applying filter if given
            if filter:
                mask = filter.mask(q, a)
                mask = mask.to(device)
                scores = scores + mask
            a = a.view(-1, 1)
            #calculating indices for sorting...
            _, _indices = torch.sort(scores, dim = 1, descending=True) #! changed: scores goes big!
            #adding index places... +1 very important for correct positioning!!!
            # 1/ for reciprocal!!!
            mean += torch.sum(1/(1+torch.eq(_indices+plus,a).nonzero()[:, 1])).item()
        return mean/(n_queries)
    
#! NEEDS FIXING (NEXT NEXT GIT PUSH)
def NDCG(train: Dataset, val: Dataset, test: Dataset, model: torch.nn.Module)->float:
    '''
    NDCG calculates overall how a query ranks all answers and normalizes
    them using the best possible ranking. (all correct answers first)

    could include an @N mechanism...
    '''
    model.eval()
    #first gather all unique answers and queries...
    n_entities = model.num_entities
    dict_ = dict()
    q_dict = dict()
    train = DataLoader(train, batch_size=1)
    print('Gathering q, a pairs...')
    for q, a in train:
        q_hash = q.hash.item()
        a = a.item()
        if q_hash not in dict_:
            dict_[q_hash] = {a}
        else:
            dict_[q_hash].add(a)
        if not (q_hash in q_dict):
            q_dict[q_hash] = q
    test = DataLoader(test, batch_size=1)
    for q, a in test:
        q_hash = q.hash.item()
        a = a.item()
        if q_hash not in dict_:
            dict_[q_hash] = {a}
        else:
            dict_[q_hash].add(a)
        if not (q_hash in q_dict):
            q_dict[q_hash] = q
    val = DataLoader(val, batch_size=1)
    for q, a in val:
        q_hash = q.hash.item()
        a = a.item()
        if q_hash not in dict_:
            dict_[q_hash] = {a}
        else:
            dict_[q_hash].add(a)
        if not (q_hash in q_dict):
            q_dict[q_hash] = q
    print('Done!')
    #now start calculating...
    #unfortunately only with batch = 1
    n = 0
    with torch.no_grad():
        ndcg = 0.
        # the discounted gains to be masked over!
        dcg = 1/torch.log2(torch.arange(1, 1+n_entities)+1)
        for q_hash in tqdm(dict_):
            # find query
            q = q_dict[q_hash]
            #extract all answers
            as_ = dict_[q_hash]
            n_a = len(as_)
            #predict scores...
            scores = model.evaluate(q, torch.arange(1, n_entities + 1).expand(1, -1), unsqueeze=True)
            #calculating indices for sorting...
            _, _indices = torch.sort(scores, dim = 1, descending=True)
            # cast as tensor (position between them not relevant (set))
            as_ = torch.Tensor(list(as_)).type(torch.int64)
            as_ = _indices.reshape(-1)[as_-1] # as_ is plus one
            mas = torch.where(torch.isin(torch.arange(1, 1+n_entities),as_) , 1, 0).reshape(1, -1)
            mbc = torch.tensor(([1]*n_a)+([0]*(n_entities-n_a))).reshape(1, -1)
            # ndcg per query
            ndcg += (torch.sum(mas*dcg)/torch.sum(mbc*dcg)).item()

    return ndcg/(len(dict_))
