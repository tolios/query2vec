from __future__ import annotations
import random
import os
import ast 
import argparse
import json
from tqdm import tqdm
import torch
from torch_geometric.data import  Data, Dataset
from form import hashQuery, structure, DEPTH_DICT

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
    # with hash
    q_hash = hashQuery(query)
    # structure
    q_struct = structure(query)
    # depth
    q_depth = DEPTH_DICT[q_struct]

    return Data(
        x=x, 
        edge_index=edge_index, 
        edge_attr=edge_attr,
        hash=q_hash,
        structure=q_struct, 
        depth=q_depth
    )

class qa_dataset(Dataset):
    '''
        This Dataset prepares query and answer data,
    to be processed for a graph neural network.
    '''
    def __init__(self, query_path: str):
        super().__init__()
        self.query_path = query_path
        self.qas, self.ids = self.extract_qas()
    def len(self) -> int:
        return len(self.ids)
    def __getitem__(self, idx):
        return self.get(idx)  # type: ignore
    def get(self, idx):
        i = self.ids[idx]
        q, ans, locid = self.qas[i]
        answer = ans[idx - locid]

        return (q, answer)

    def extract_qas(self)->list:
        qas = dict()
        ids = dict()
        print('Extracting qa data...')
        global_id = 0
        locid = 0
        with open(self.query_path, 'r') as f:
            for i, line in enumerate(f):
                q, ans = ast.literal_eval(line) #get query and total answers
                qas[i] = (query2graph(q), ans, locid)
                locid += len(ans)
                for _ in ans:
                    ids[global_id] = i
                    global_id += 1
        print('Done!')
        return qas, ids

class connections():
    '''
        Custom class containing connections of a graph structure.
    It provides sampling methods. As well as other utilities...
    To be used for extraction of data, as well as query & answer generation (sampling).

    version: 2.0: calculate all answers
    version: 1.1: added inverses | r, r/-1 !!!

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
        self._2relationships, self._2heads, self._2tails = self.extract_connections(path, self.entity2id, self.relationship2id, add_inverse = add_inverse)
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
        _2tails = dict()

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
                #2tails
                if (h, r) not in _2tails:
                    _2tails[(h, r)] = set()
                    _2tails[(h, r)].add(t)
                else:
                    _2tails[(h, r)].add(t)
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
                    #2tails
                    if (t, r_inv) not in _2tails:
                        _2tails[(t, r_inv)] = set()
                        _2tails[(t, r_inv)].add(h)
                    else:
                        _2tails[(t, r_inv)].add(h)
        #return details
        return _2relationships, _2heads, _2tails

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
        
        for key in other._2tails:
            if key in self._2tails:
                self._2tails[key] = self._2tails[key] | other._2tails[key]
            else:
                self._2tails[key] = other._2tails[key]


    def sample_link(self, t: int)->tuple:
        #* t as in tail of directed edge...
        r = random.choice(list(self._2relationships[t]))
        h = random.choice(list(self._2heads[(r, t)]))
        return h, r

    def random_answer(self)->int:
        #*  This method outputs a random possible entity id as answer.
        #* It must be a tail of a known link!
        return random.choice(list(self._2relationships))

    def sample_qa(self, 
        num_edges: int = 2, other: connections|None = None, 
        uniques: set = set(), structures = ["2p", "2i", "3p", "3i", "ip", "pi", "not_implemented"])->tuple:
        #*  Samples a query (as a dag) and answer, with number of edges as num_edges, as well as its answers
        #* Sometimes, it fails and produces a rejected query, so it outputs ([], [], uniques).

        query = []
        answers = []
        known_link = set()
        #pick a start!
        t = self.random_answer()
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
                return [], answers, uniques
            if (h, r, t) in known_link:
                #since it generated the same link twice, simply reject... (guarrantees n edges)
                return [], answers, uniques
            else:
                known_link.add((h, r, t))
            if h in variables:
                #check if correct order, so as to avoid cycles...
                order_h = int(variables[h][1:])
                order_t = int(variables[t][1:]) if t in variables else vars + 1
                #must order_h > order_t, else cycle!!!
                if order_h <= order_t:
                    #cycle, therefore reject...
                    return [], answers, uniques
            #name new variable for each new entity as tail!
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
                    return [], answers, uniques
        #check if query contains one of the required links, else return malformed...
        if not has_required:
            return [], answers, uniques
        #Replace with variables!
        true_query = []
        for link in query:
            h, r, t = link
            h = variables[h] if h in variables else h
            t = variables[t] if t in variables else t
            true_query.append([h, r, t])
        
        #check if it has degenerate join -r->n-r^-1->
        if connections.detect_inverse_degenerate(true_query):
            return [], answers, uniques

        # find structure
        q_struct = structure(true_query)

        if not (q_struct in structures):
            return [], answers, uniques

        # find if unique query
        q_hash = hashQuery(true_query)

        # unique representation...
        if q_hash in uniques:
            return [], answers, uniques # rejected because query has been sampled already!
        else:
            # first generate all answers!
            answers = self.get_answers_from_query(true_query, other=other)
            if not answers:
                # a highly unlikely case where train, val, test might have common answers for a question (so the filter works for all the produced answers for other)
                return [], answers, uniques
            uniques.add(q_hash)

        return true_query, answers, uniques

    def generate_1hops(self):
        #* This method yields all 1 hop queries and their answers!
        for key, value in self._2tails.items():
            yield ([[key[0], key[1], '_1']], list(value))

    def write_qas(self, qa_name, other: connections|None = None,
            query_orders: list = [[(2, 6000)]],
            structures = [["1p", "2p", "2i", "3p", "3i", "ip", "pi", "not_implemented"]], 
            tot_tries = 10e7)->None:
            #*  This is the main method of the class. 
            #* It writes all the contents in a specified file
            #* First we combine with other, if it exist, then 
            #* we produce queries and answers!

            if len(query_orders) != len(structures):
                print('query orders and structures should be lists of same length')
                raise

            if other:
                self.combine(other)

            n = 0

            for query_orders_, structures_ in zip(query_orders, structures):

                n += 1

                qa_path = qa_name+f"_{n}.txt"
                _, name = os.path.split(qa_path)

                print(f"Making of {name}...")

                with open(qa_path, 'w') as f:
                    # query_orders, structures must be the same length...
                    
                    #1st the 1hops ... if requested!
                    if '1p' in structures_:
                        if other:
                            #If specified, we should use the others
                            #1 hops....(which contain the needed links!)
                            #Used to generate val, test data...
                            for qa in tqdm(other.generate_1hops(), desc=f'Generating 1hops inside {name}'):
                                f.write(str(qa)+'\n')
                        else:
                            #Used to generate train data...
                            for qa in tqdm(self.generate_1hops(), desc=f'Generating 1hops inside {name}'):
                                f.write(str(qa)+'\n')
                    #now we should use the sampling method for the rest...
                    for  orders_, structure in zip(query_orders_, structures_):
                        num_edges, num_queries = orders_
                        uniques = set()
                        tries = 1 #used for a cuttoff if the program takes too long?!
                        pbar = tqdm(total = num_queries, desc = f"Generating queries with #edges = {num_edges} inside {name}")
                        while len(uniques) < num_queries and tries < tot_tries:
                            #generate query and answers while updating uniques
                            q, ans, uniques = self.sample_qa(num_edges=num_edges, 
                                                other=other, uniques=uniques, structures=structure)
                            #write (q, a) if q non empty
                            if q:
                                f.write(str((q, ans))+'\n')
                                pbar.update(1)
                            else:
                                tries += 1
                        pbar.close()
                        if tries > tot_tries:
                            print(f'Exceeted tot_tries = {tot_tries}')
                
                print(f"Finished {name} !")

    def get_answers_from_query(self, query, other: connections|None = None):
        #NOTE - Receives query and produces all available answers.
        #if other given, all answers must have used at least one
        #link from other. (Used for val, test)
        variables = dict()
        required = dict()
        for _, _, t in query:
            if t in required:
                required[t] += 1
            else:
                required[t] = 1
        while query:
            new_query = []
            not_now = set() #gets reset every time
            for link in query:
                h, r, t = link
                if isinstance(h, int):
                    # since anchor node simply get all variable answers...
                    if t in variables:
                        variables[t] = self.intersect_variables(variables[t], self.get_tails_from_anchor(h, r, other=other))
                    else:
                        variables[t] = self.get_tails_from_anchor(h, r, other=other)
                    #remove required contribution
                    required[t] = required[t] - 1
                    if required[t] == 0:
                        not_now.add(t)
                else:
                    # if h has received values and they are all of them and its time
                    if (h in variables) and (required[h] == 0) and (not (h in not_now)):
                        # get answers
                        if t in variables:
                            variables[t] = self.intersect_variables(variables[t],self.get_tails_from_vars(variables[h], r, other=other))
                        else:
                            variables[t] = self.get_tails_from_vars(variables[h], r, other=other)
                        #remove required contribution
                        required[t] = required[t] - 1
                        if required[t] == 0:
                            not_now.add(t)
                    else:
                        new_query.append([h, r, t]) #keep for next round
            query = new_query
        
        # keep only answers with base info...
        not_other = {i for i, in_other in variables["_1"] if not in_other}

        if other:
            # impressively we need to filter out the possibility that an answer is produced
            # both by other and without!!!
            return list({i for i, in_other in variables["_1"] if in_other} - not_other)
        else:
            return list(not_other)
    
    def get_tails_from_vars(self, hs, r, other: connections|None = None):
        tails = set()
        for h, in_other in hs:
            tails = tails | {(i, False if not other else (other.__contains__((h, r, i)) or in_other)) for i in self._2tails.get((h, r), set())}
        return tails
    
    def get_tails_from_anchor(self, anchor, r,  other: connections|None = None):
        # gets all tails, and adds information if its from other or not...
        return {(i, False if not other else other.__contains__((anchor, r, i))) for i in self._2tails.get((anchor, r), set())}

    @staticmethod
    def intersect_variables(left, right):
        result = set()
        #iterate with the lower of the two
        if len(left) > len(right):
            left, right = right, left
        for v, in_other in left:
            # if exists in left acts with an or for the bool parameter
            # we describe if v has info that has used at east one from the other links (the missing)
            if (v, True) in right:
                result.add((v, True))
            if (v, False) in right:
                result.add((v, in_other))
        
        return result

    @staticmethod
    def detect_inverse_degenerate(query: list):
        # detect -r->n-r^-1-> connections to throw out!
        # extract tail, head with relation connections
        h2r = dict()
        t2r = dict()
        for h, r, t in query:
            if not isinstance(h, int):
                if not (h in h2r):
                    h2r[h] = []
                h2r[h].append(r)
            if not (t in t2r):
                t2r[t] = []
            t2r[t].append(r)
            
        for node in h2r:
            hrs = h2r[node]
            trs = t2r[node]

            for hr in hrs:
                for tr in trs:
                    # detect inverse join with regular using node
                    if hr % 2 == 0 and tr % 2 == 1 and (tr - hr == 1):
                        return True
                    if tr % 2 == 0 and hr % 2 == 1 and (hr - tr == 1):
                        return True
        return False



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
                        type = str, default='[[(2, 60000)]]',
                help='''
                    Besides 1hop queries this argument can determine how many qa pairs that have DAGs with a specified num_edges. 
                    Expects a string of the form "[(num_edges, num_queries), ...]. num_edges: int >= 2"''')
    parser.add_argument("--val_query_orders",
                        type = str, default='[[(2, 60000)]]',
                help='''
                    Besides 1hop queries this argument can determine how many qa pairs that have DAGs with a specified num_edges. 
                    Expects a string of the form "[(num_edges, num_queries), ...]. num_edges: int >= 2"''')
    parser.add_argument("--test_query_orders",
                        type = str, default='[[(2, 60000)]]',
                help='''
                    Besides 1hop queries this argument can determine how many qa pairs that have DAGs with a specified num_edges. 
                    Expects a string of the form "[(num_edges, num_queries), ...]. num_edges: int >= 2"''')
    parser.add_argument("--add_inverse",
                        type = bool, default=False,
                help='''Add inverses for all relations in the KG, doubling all relations represented!''')
    parser.add_argument("--include_train",
                    type = str, default='[["1p", "2p", "2i", "3p", "3i", "ip", "pi", "not_implemented"]]',
                help='''
                    This argument decides which of the query structures will be included in train''')
    parser.add_argument("--include_val",
                    type = str, default='[["1p", "2p", "2i", "3p", "3i", "ip", "pi", "not_implemented"]]',
                help='''
                    This argument decides which of the query structures will be included in val''')
    parser.add_argument("--include_test",
                    type = str, default='[["1p", "2p", "2i", "3p", "3i", "ip", "pi", "not_implemented"]]',
                help='''
                    This argument decides which of the query structures will be included in test''')

    args = parser.parse_args()

    #directory where triplets are stored...
    path=os.path.dirname(args.train_path)

    #mkdir that will house queries!
    qa_dir = path+'/'+args.qa_folder
    os.mkdir(qa_dir)

    #Make queries...
    #First extract connections and id mappings...
    train = connections(args.train_path, start=args.start, add_inverse=args.add_inverse)
    val = connections(args.val_path, entity2id = train.entity2id, relationship2id = train.relationship2id, start=args.start, add_inverse=args.add_inverse) #TODO - UNKNOWN IF ITS ACTUALLY ALLOWED
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
    # query structures
    include_train = ast.literal_eval(args.include_train)
    include_val = ast.literal_eval(args.include_train)
    include_test = ast.literal_eval(args.include_test)

    #Making query files...
    train.write_qas(qa_dir+'/train_qa', query_orders=train_query_orders, structures=include_train)
    train.write_qas(qa_dir+'/val_qa', other = val, query_orders=val_query_orders, structures=include_val) #combines with val and keeps one edge (atleast) 
    train.write_qas(qa_dir+'/test_qa', other = test, query_orders=test_query_orders, structures=include_test) #combines with test and keeps one edge (atleast)

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
        'add_inverse': args.add_inverse,
        'include_train': args.include_train,
        'include_val': args.include_val,
        'include_test': args.include_test
    }

    with open(qa_dir+'/info.json', 'w') as f:
        json.dump(info, f)
