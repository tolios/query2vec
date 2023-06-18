from __future__ import annotations
import random
import os
import ast 
import argparse
import json
from tqdm import tqdm
import torch
from torch_geometric.data import  Data, Dataset
from hash import hashQuery

def query2graph(query: list)->Data:
    #*  Receives a query in form of list of triples and 
    #* returns a pytorch geometric data class, 
    #* which represents said DAG form of the given query.

    x = list()
    edge_index = list()
    edge_attr = list()
    #dict for variables...
    variables = dict()
    n = 0
    h_index = 0
    t_index = 0

    for h, r, t in query:
        #parse triplet...
        if not isinstance(h, int):
            if h not in variables:
                variables[h] = n
                x.append(0)
                n += 1
            h_index = variables[h]
            #change to 0
            h = 0
        else:
            #since new value...
            h_index = n
            x.append(h)
            n += 1
        if not isinstance(t, int):
            if t not in variables:
                variables[t] = n
                x.append(0)
                n += 1
            t_index = variables[t]
            t = 0
        else:
            #since new value...
            t_index = n
            x.append(t)
            n += 1

        edge_index.append([h_index, t_index])
        edge_attr.append(r)

    #* Preparing the right format...
    x = torch.LongTensor(x).unsqueeze(-1)
    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.LongTensor(edge_attr).unsqueeze(-1)

    #* Return Data...
    #with hash
    q_hash = hashQuery(query)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, hash=q_hash)

class qa_dataset(Dataset):
    '''
        This Dataset prepares query and answer data,
    to be processed for a graph neural network.
    '''
    def __init__(self, query_path: str):
        super().__init__()
        self.query_path = query_path
        self.qas = self.extract_qas()
    def len(self) -> int:
        return len(self.qas)
    def __getitem__(self, idx):
        return self.get(idx)  # type: ignore
    def get(self, idx):
        return self.qas[idx]
    def extract_qas(self)->list:
        qas = []
        print('Extracting qa data...')
        with open(self.query_path, 'r') as f:
            for line in f:
                q, a = ast.literal_eval(line)
                qas.append((query2graph(q), a))
        print('Done!')
        return qas

class connections():
    '''
        Custom class containing connections of a graph structure.
    It provides sampling methods. As well as other utilities...
    To be used for extraction of data, as well as query & answer generation (sampling).

    version: 1.1: added inverses | r, r/-1 !!!

    #! Needs better sampler?
    #! having the same entity, different relation!? A relativeOf B, A neighbourOf B (makes sense?!)
    ([[3366, 325, '_1'], [3366, 77, '_1']], 3365)
    ([[400, 9, '_1'], [118, 9, '_1']], 9678)
    '''
    def __init__(self, path: str, 
                entity2id: dict = dict(), 
                relationship2id: dict = dict(), 
                start: int = 1,
                add_inverse=False):
        self.path = path
        if not (entity2id and relationship2id):
            #if any one of them is not defined...
            del entity2id, relationship2id
            self.entity2id, self.relationship2id = self.extract_mappings(path, start = start, add_inverse = add_inverse)
        else:
            #use given mappings (useful for val and test data...)
            self.entity2id, self.relationship2id = entity2id, relationship2id
        self._2relationships, self._2heads = self.extract_connections(path, self.entity2id, self.relationship2id, add_inverse = add_inverse)
        self.add_inverse = add_inverse

    def __contains__(self, key: tuple)->bool:
        #* key is a triplet (h, r, t)
        h, r, t = key
        if t in self._2relationships:
            if r in self._2relationships[t]:
                if h in self._2heads[(r, t)]:
                    #it is one of the required links!
                    return True

        return False

    def extract_mappings(self, path: str, start: int = 1, add_inverse: bool = False)->tuple:
        '''
        Receives a triple file (a training file, that has atleast one of all total entities & relationships)...
        Returns a tuple, with two dictionaries that map entities & relationships to corresponding ids!
        start is used to offset the ids of entities so as the first start-1 ids are unused! For example start=1, we have no id = 0...

        Has the ability to add inverses, if r is the relation, r/-1 is the defined inverse!
        '''
        unique_objects = dict()
        unique_relationships = dict()

        id_object = start #offset
        id_relationship = 0 #no offset used for relationship ids... (YET?)

        with open(path, 'r') as file:
            for line in file:
                #tab separated values!!!
                h, r, t = line.split('\t')
                #remove \n
                t = t[:-1]
                #we will encode the nodes and edges with unique integers!
                #this will match with the embedding Tensors that are used to contain
                #embeddings for our objects!
                if h not in unique_objects:
                    unique_objects[h] = id_object
                    id_object += 1
                if t not in unique_objects:
                    unique_objects[t] = id_object
                    id_object += 1
                if r not in unique_relationships:
                    unique_relationships[r] = id_relationship
                    id_relationship += 1
                if add_inverse:
                    if r+"/-1" not in unique_relationships:
                        unique_relationships[r+"/-1"] = id_relationship
                        id_relationship += 1

        return unique_objects, unique_relationships

    
    def extract_connections(self, triplet_file: str, entity2id:dict, relationship2id:dict, add_inverse: bool = False)->tuple:
        _2relationships = dict()
        _2heads = dict()

        with open(triplet_file, 'r') as f:
            for line in f:
                h, r, t = line[:-1].split('\t')
                if h in entity2id and t in entity2id and r in relationship2id:
                    if add_inverse:
                        r_inv = relationship2id[r+"/-1"]
                    h, r, t = entity2id[h], relationship2id[r], entity2id[t]
                else:
                    continue
                #2relationships
                if t not in _2relationships:
                    _2relationships[t] = set()
                    _2relationships[t].add(r)
                else:
                    _2relationships[t].add(r)
                #2heads
                if (r, t) not in _2heads:
                    _2heads[(r,t)] = set()
                    _2heads[(r,t)].add(h)
                else:
                    _2heads[(r,t)].add(h)
                #inverses!
                if add_inverse:
                    #2relationships
                    if h not in _2relationships:
                        _2relationships[h] = set()
                        _2relationships[h].add(r_inv)
                    else:
                        _2relationships[h].add(r_inv)
                    #2heads
                    if (r_inv, h) not in _2heads:
                        _2heads[(r_inv,h)] = set()
                        _2heads[(r_inv,h)].add(t)
                    else:
                        _2heads[(r_inv,h)].add(t)
        #return details
        return _2relationships, _2heads

    def combine(self, other: connections)->None:
        #* This function combines the knowledge of two graphs into one.
        #*This means that self now represents the combined graph!!!
        for key in other._2relationships:
            if key in self._2relationships:
                self._2relationships[key] = self._2relationships[key] | other._2relationships[key]
            else:
                self._2relationships[key] = other._2relationships[key]

        for key in other._2heads:
            if key in self._2heads:
                self._2heads[key] = self._2heads[key] | other._2heads[key]
            else:
                self._2heads[key] = other._2heads[key]


    def sample_link(self, t: int)->tuple:
        #* t as in tail of directed edge...
        r = random.choice(list(self._2relationships[t]))
        h = random.choice(list(self._2heads[(r, t)]))
        return h, r

    def random_answer(self)->int:
        #*  This method outputs a random possible entity id as answer.
        #* It must be a tail of a known link!
        return random.choice(list(self._2relationships))

    def sample_qa(self, num_edges: int = 2, other: connections|None = None)->tuple:
        #*  Samples a query (as a dag) and answer, with number of edges as num_edges.
        #* Sometimes, it fails and produces a rejected query, so it outputs ([], answer).

        query = []
        known_link = set()
        #pick a start!
        answer = self.random_answer()
        t = answer
        visited_nodes = [t] #nodes known to the graph!
        variables = {t: '_1'} #nodes that have incoming nodes! -e->(v) 
        vars = 1 #num of variables! #_num important to keep order for dag!
        #has required automatically true if we have no required links...
        has_required = not other
        for _ in range(num_edges):
            #sample a link...
            h, r = self.sample_link(t)
            #rejections...
            if h == t:
                #self loops are killed on the spot...
                return [], answer
            if (h, r, t) in known_link:
                #since it generated the same link twice, simply reject... (guarrantees n edges)
                return [], answer
            else:
                known_link.add((h, r, t))
            if h in variables:
                #check if correct order, so as to avoid cycles...
                order_h = int(variables[h][1:])
                order_t = int(variables[t][1:]) if t in variables else vars + 1
                #must order_h > order_t, else cycle!!!
                if order_h <= order_t:
                    #cycle, therefore reject...
                    return [], answer
            #Name new variable for each new entity as tail!
            if t not in variables:
                vars += 1
                variables[t] = f'_{vars}'
            #check if link is from the required ones (other)!
            #if it has even one, dont bother...
            if not has_required:
                if (h, r, t) in other:
                    has_required = True
            #add to visited...
            visited_nodes.append(h)
            #add link to query...
            query.append([h, r, t])
            #decide new tail to generate link...
            t = random.choice(visited_nodes)
            #check if new tail is a root of the actual graph!
            #if it is, we cannot use it as a new tail...
            if t not in self._2relationships:
                #if we do have enough, keep. Else reject!
                if len(query) != num_edges:
                    return [], answer
        #check if query contains one of the required links, else return malformed...
        if not has_required:
            return [], answer
        #Replace with variables!
        true_query = []
        for link in query:
            h, r, t = link
            h = variables[h] if h in variables else h
            t = variables[t] if t in variables else t
            true_query.append([h, r, t])

        return true_query, answer

    def generate_1hops(self):
        #* This method yields all 1 hop queries and their answers!
        for key in self._2heads:
            for h in self._2heads[key]:
                yield ([[h, key[0], '_1']], key[1])

    def write_qas(self, qa_path:str, other: connections|None = None,
            query_orders: list = list(), tot_tries = 10e7)->None:
            #*  This is the main method of the class. 
            #* It writes all the contents in a specified file
            #* First we combine with other, if it exist, then 
            #* we produce queries and answers!

            _, name = os.path.split(qa_path)

            if other:
                self.combine(other)

            with open(qa_path, 'w') as f:
                #1st the 1hops ...
                if other:
                    #If specified, we should use the others
                    #1 hops....(which contain the needed links!)
                    #Used to generate val, test data...
                    for qa in tqdm(other.generate_1hops(), desc=f'Generating 1hops inside {name}'):
                        f.write(str(qa)+'\n')
                else:
                    #Used to generate train data...
                    for qa in self.generate_1hops():
                        f.write(str(qa)+'\n')
                #now we should use the sampling method for the rest...
                for num_edges, num_queries in query_orders:
                    uniques = set()
                    tries = 1 #used for a cuttoff if the program takes too long?!
                    pbar = tqdm(total = num_queries, desc = f"Generating queries with #edges = {num_edges} inside {name}")
                    while len(uniques) < num_queries and tries < tot_tries:
                        #generate query...
                        qa = self.sample_qa(num_edges=num_edges, other=other)
                        #check if not rejected...
                        if qa[0]:
                            qastr = str(qa)
                            if qastr not in uniques:
                                uniques.add(qastr)
                                f.write(qastr+'\n')
                                pbar.update(1)
                        tries += 1
                    pbar.close()
                    if tries > tot_tries:
                        print(f'Exceeted tot_tries = {tot_tries}')

def corrupted_answer(num_entities, batch_size, num_negs=1,start = 1):
    return torch.randint(high=num_entities, size=(batch_size[0], num_negs))+start

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Parsing triplets file...')

    #requirement arguments...
    parser.add_argument("train_path",
                        type=str, help="Path of file where training triplets are saved...")

    parser.add_argument("val_path",
                        type=str, help="Path of file where validation triplets are saved...")

    parser.add_argument("test_path",
                        type=str, help="Path of file where testing triplets are saved...")

    parser.add_argument("--qa_folder",
                        type = str, default='qa', help="QA folder name...")
    
    parser.add_argument("--start",
                        type = int, default=1, help="Determines the starting id of entities!")

    parser.add_argument("--train_query_orders",
                        type = str, default='[(2, 60000)]',
                help='''
                    Besides 1hop queries this argument can determine how many qa pairs that have DAGs with a specified num_edges. 
                    Expects a string of the form "[(num_edges, num_queries), ...]. num_edges: int >= 2"''')
    parser.add_argument("--val_query_orders",
                        type = str, default='[(2, 60000)]',
                help='''
                    Besides 1hop queries this argument can determine how many qa pairs that have DAGs with a specified num_edges. 
                    Expects a string of the form "[(num_edges, num_queries), ...]. num_edges: int >= 2"''')
    parser.add_argument("--test_query_orders",
                        type = str, default='[(2, 60000)]',
                help='''
                    Besides 1hop queries this argument can determine how many qa pairs that have DAGs with a specified num_edges. 
                    Expects a string of the form "[(num_edges, num_queries), ...]. num_edges: int >= 2"''')
    parser.add_argument("--add_inverse",
                        type = bool, default=False,
                help='''Add inverses for all relations in the KG, doubling all relations represented!''')

    args = parser.parse_args()

    #directory where triplets are stored...
    path=os.path.dirname(args.train_path)

    #mkdir that will house queries!
    qa_dir = path+'/'+args.qa_folder
    os.mkdir(qa_dir)

    #Make queries...
    #First extract connections and id mappings...
    train = connections(args.train_path, start=args.start, add_inverse=args.add_inverse)
    val = connections(args.val_path, entity2id = train.entity2id, relationship2id = train.relationship2id, start=args.start, add_inverse=args.add_inverse)
    test = connections(args.test_path, entity2id = train.entity2id, relationship2id = train.relationship2id, start=args.start, add_inverse=args.add_inverse)

    #save mappings to the qa folder!
    with open(qa_dir+'/entity2id.json', 'w') as f:
        json.dump(train.entity2id, f)
    with open(qa_dir+'/relationship2id.json', 'w') as f:
        json.dump(train.relationship2id, f)

    import ast
    #transforms string to correct list of tuples form...
    train_query_orders = ast.literal_eval(args.train_query_orders)
    val_query_orders = ast.literal_eval(args.val_query_orders)
    test_query_orders = ast.literal_eval(args.test_query_orders)

    #Making query files...
    train.write_qas(qa_dir+'/train_qa.txt', query_orders=train_query_orders)
    train.write_qas(qa_dir+'/val_qa.txt', other = val, query_orders=val_query_orders) #combines with val and keeps one edge (atleast) 
    train.write_qas(qa_dir+'/test_qa.txt', other = test, query_orders=test_query_orders) #combines with test and keeps one edge (atleast)

    #Making an info json file...
    info = {
        'train_path': args.train_path,
        'val_path': args.val_path,
        'test_path': args.test_path,
        'num_entities': len(train.entity2id),
        'num_relationships': len(train.relationship2id), 
        'train_query_orders': args.train_query_orders,
        'val_query_orders': args.val_query_orders,
        'test_query_orders': args.test_query_orders,
        'add_inverse': args.add_inverse
    }

    with open(qa_dir+'/info.json', 'w') as f:
        json.dump(info, f)
