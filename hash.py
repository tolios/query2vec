'''
        This file contains a possible hashing of query dags algorithm so as to
    distinguish between actually different dags, and isomorphic dags!

    #! NOT PROVEN !!!
'''

def parse_graphs(graph:list[tuple])->dict[str, set]:
    # parses the graph and produces dict with all connections!
    paths = dict()
    for triple in graph:
        h, r, t = triple
        if t not in paths:
            paths[t] = set()
        paths[t].add((h, r))
    return paths

def hash_variable(paths: dict[str, set], tail: str):
    ''' 
        Given a tail, it finds all heads connected with it,
    creates a list of tuples, where each tuple has the hash 
    of the head, and the relation that connects it with the tail.
    Then we sort the aforementioned list of tuples by the hashed
    heads, then the relations.
        If the head is a variable (meaning a string) we call again
    the hash_variable algorithm. If it is an integer, therefore an 
    anchor, we simply return the head it self!
        When we have sorted the list, we return the hash!
    '''
    # Now we iterate the connected links of the tail to create: (hash head, relation)
    connections = []
    for h, r in paths[tail]:
        if not isinstance(h, int):
            # since a variable!
            h = hash_variable(paths, h)
        connections.append((h, r))
    #sort connections first with hashed head then the relations...
    connections.sort(key=lambda t: (t[0], t[1]))

    return hash(tuple(connections))

def hashQuery(query: list)->int:
    '''
        This algorithm attempts to produce a hash of a query,
    where two equivalent queries will have the same hash and 
    two different queries will have a different hash.

    Due to the hash_variable algorithm, it boils down to a
    DFS, with sorting on local connections!

    # VERY SIMILAR TO MERKLE TREES (MERKLE DAGS?)
    #! UNPROVEN FOR GENERAL DAG QUERIES (Might be provable for Trees)
    '''
    # first extract the paths
    paths = parse_graphs(query)
    # hashes the answer node, which also hashes the query
    return hash_variable(paths, "_1")
