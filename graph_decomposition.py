import networkx as nx
import operator


def check_duplicates(tpl):
    if tpl[0] != tpl[1]:
        return True
    else:
        return False

def filter_duplicates(dct) -> dict:
    """deletes dictionary keys with duplicate tuples"""
    dct_filtered = dict()

    for (key, value) in dct.items():
        if check_duplicates(key):
            dct_filtered[key] = value
    else:
        pass

    return dct_filtered


# func that returns n-edge with max weight
def max_weight(graph):
    """returns edge with max weight"""
    labels = nx.get_edge_attributes(graph, 'weight')
    labels_filtered = filter_duplicates(labels)
    max_edge = max(labels_filtered.items(), key=operator.itemgetter(1))[0]
    return max_edge


# func that finds maximun simplex of graph
def max_simplex(graph):
    pass

