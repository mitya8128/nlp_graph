import networkx as nx
from networkx.algorithms.approximation import clique
import operator
from utils import*


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
def max_weight(graph) -> list:
    """returns edge with max weight"""
    labels = nx.get_edge_attributes(graph, 'weight')
    labels_filtered = filter_duplicates(labels)
    max_edge = max(labels_filtered.items(), key=operator.itemgetter(1))[0]
    return max_edge


# func that finds maximum simplex of graph
def max_simplex(graph):
    pass


def sorted_weights(graph) -> list:
    """returns list with edges sorted by their weights"""
    labels = nx.get_edge_attributes(graph, 'weight')
    labels_filtered = filter_duplicates(labels)
    sorted_edges = sorted(labels_filtered.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_edges


def find_cliques_all(graph) -> list:
    """simple interface for nx function
    returns list of all cliques"""
    return list(nx.algorithms.clique.find_cliques(graph))


def find_max_clique(graph) -> set:
    """simple interface for nx function
        returns max clique"""
    return clique.max_clique(graph)


def get_weight(labels, a, b) -> float:
    """get weight of specific edge"""
    tpl = (a,b)
    return labels[tpl]


def clique_weights(clique):
    """returns sum of all edges of clique"""
    vec_clique = []
    for i in clique:
        vec_word = []
        for j in clique:
            vec = cosine(model[i], model[j])
            vec_word.append(vec)
        vec_clique.append(vec_word)

    sum_all = []
    for i in range(len(vec_clique)):
        sum_i = sum(vec_clique[i])
        sum_all.append(sum_i)

    return sum(sum_all)