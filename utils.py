import re
import random

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import networkx as nx
import nltk
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters
import stopwordsiso as stopwords
from string import punctuation
import matplotlib.pyplot as plt


morph = MorphAnalyzer()
to_ud = converters.converter('opencorpora-int', 'ud20')

stops = stopwords.stopwords("ru")
added_stops = {'весь', 'это', 'наш', 'оно', 'итак', 'т.п', 'т.е', 'мало', 'меньше', 'ещё', 'слишком', 'также',
               'ваш', 'б', 'хм', 'который', 'свой', 'не', 'мочь', 'однако', 'очень', 'благодаря', 'кроме', 'вся',
               'какие', 'ru', 'en', 'млрд', 'млн', 'нет', 'этот', 'мной', 'дело', 'был', 'долго', 'наша', 'самих',
               'миллионов', 'самых', 'ост', 'ст', 'д', 'проспект', 'компания', 'компании', 'компанию', 'компанией',
               'компаниям', 'e-mail', 'шаг', 'ул', 'rus', 'eng', 'проезд', 'площадь', 'cookies', 'куки', 'кг', 'xl',
               'rss', 'amp', ';amp', 'pdf', 'doc', 'txt', 'docx', 'i', 'id',
               'бывший'}

stops = stops.union(added_stops)
punct = punctuation + '«»—…“”*№–'

model_path = '/home/mitya/PycharmProjects/nlp_graph/model.bin'
model = KeyedVectors.load_word2vec_format(model_path, binary=True)


def pymorphy_tagger(text):
    text = text.replace('[', ' ').replace(']', ' ')
    parsed = []
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        word = word.strip(punct)
        if (word not in stops) and (word not in punct) and (
                re.sub(r'[{}]+'.format(punct), '', word).isdigit() is False) and (word != 'nan'):
            lemma = str(morph.parse(word)[0].normal_form)
            pos = to_ud(str(morph.parse(word)[0].tag.POS)).split()[0]
            word_with_tag = lemma + '_' + pos
            parsed.append(word_with_tag)
    return ' '.join(parsed)


def cosine(a, b):
    dot = np.dot(a, b.T)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos


def similar_words(text, n):
    """return n most similar words in models dictionary"""

    lst = model.most_similar(pymorphy_tagger(text), topn=n)

    return lst


def diff(text1, text2):
    return cosine(model[text1], model[text2])


def vertices(text, n):
    """return list of vertices based on similar_words function"""

    vertices = similar_words(text, n)
    vertices_list = [vertices[0] for vertices in vertices]

    return vertices_list


def adjacency_mat(vertices_list):
    """make matrix of distances between words"""

    n = len(vertices_list)
    adj_mat = []
    for i in vertices_list:
        for j in vertices_list:
            adj_vec = []
            vec = cosine(vectorize_word(i), vectorize_word(j))
            adj_vec.append(vec)
            adj_mat.append(adj_vec)

    return np.array(adj_mat).reshape(n, n)


def make_graph(mat, vertices_list, th):
    """make graph with edges between vertices based on adjacency_mat function"""

    G = nx.from_numpy_matrix(mat)
    mapping = dict(zip(G, vertices_list))
    H = nx.relabel_nodes(G, mapping)
    labels = nx.get_edge_attributes(H, 'weight')
    labels_filtered = dict()

    for (key, value) in labels.items():
        if value <= th:
            labels_filtered[key] = value
        else:
            pass

    e = getList(labels_filtered)

    for element in e:
        H.remove_edge(*element)

    return H


def draw_graph(graph, node_size, alpha, show_weights=False):
    """draw graph: specify size of nodes and transparency of edges"""

    labels = nx.get_edge_attributes(graph, 'weight')
    pos = nx.spring_layout(graph)
    plt.figure()
    nx.draw(graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=node_size, node_color='pink', alpha=alpha,
            labels={node: node for node in graph.nodes()})

    if show_weights:
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels)
    else:
        pass

    plt.show()


def text2graph(text, th):
    """full pipeline to make graph from text"""

    text_tagged = pymorphy_tagger(clean_numbers(text))
    text_tagged = text_tagged.split()
    text_mat = adjacency_mat(text_tagged)
    graph = make_graph(text_mat, text_tagged, th)

    return graph


def getList(dct):
    list = []
    for key in dct.keys():
        list.append(key)

    return list


def clean_numbers(text):
    text = re.sub(r'[0-9]+', '', text)
    return text


def vectorize_word(word):
    """vectorize word with unknown word handler"""
    try:
        vec = model[word]
    except KeyError:
        vec = np.random.normal(0, np.sqrt(0.25), 300)

    return vec


def metric_filtration(text_mat, text_tagged):
    """filtration by metric value"""
    average_clustering = {}
    for i in np.arange(0.1, 0.9, 0.1):
        graph = make_graph(text_mat, text_tagged, i)
        average_clustering['threshold_{}'.format(i)] = nx.average_clustering(graph)
    return average_clustering


def draw_filtration_metric(average_clustering):
    """draw graph of metric filtration"""
    plt.bar(range(len(average_clustering)), list(average_clustering.values()), align='center')
    plt.xticks(range(len(average_clustering)), list(average_clustering.keys()))


def generate_random(lst):
    """generate random equal-length list from POS-tagged text"""
    newlist = []
    for i in range(len(lst)):
        element = random.choice(lst)
        newlist.append(element)
    return newlist


def select_triangles(lst, n):
    """select n-dimensional cliques"""
    triangles = []
    for i in lst:
        if len(i) == n:
            triangles.append(i)
    return triangles


def jaccard_similarity(g, h):
    """Jaccard distance between graphs"""
    i = set(g).intersection(h)
    return round(len(i) / (len(g) + len(h) - len(i)),3)


def average_distance(df, topic):
    """computes average distance between text-graphs inside topics"""
    df_topic = df[df['topic'] == topic]

    texts = df_topic['text'].tolist()

    list_graphs = []
    for i in range(len(texts)):
        graph = text2graph(texts[i], 0.1)
        list_graphs.append(graph)

    list_dist = []
    for i in list_graphs:
        for j in list_graphs:
            dist = jaccard_similarity(i, j)
            list_dist.append(dist)

    return round((sum(list_dist) / len(list_dist)), 3)