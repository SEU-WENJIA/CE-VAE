import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import scipy.io as io
import zipfile as zf


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
def load_protein():
    n = io.loadmat("data/Homo_sapiens.mat")
    return n['network'], n['group']

def load_enzyme():
    adj = sp.lil_matrix((125, 125))
    features = sp.lil_matrix((125, 1))
    for line in open("data/ENZYMES_g296.edges"):
        vals = line.split()
        x = int(vals[0]) - 2
        y = int(vals[1]) - 2
        adj[y, x] = adj[x, y] = 1
    return adj, features

def load_florida():
    adj = sp.lil_matrix((128, 128))
    features = sp.lil_matrix((128, 1))
    for line in open("data/eco-florida.edges"):
        vals = line.split()
        x = int(vals[0]) - 1
        y = int(vals[1]) - 1
        val = float(vals[2])
        adj[y, x] = adj[x, y] = val
    return adj, features

def load_brain():
    adj = sp.lil_matrix((1780, 1780))
    features = sp.lil_matrix((1780, 1))
    nums = []
    for line in open("data/bn-fly-drosophila_medulla_1.edges"):
        vals = line.split()
        x = int(vals[0]) - 1
        y = int(vals[1]) - 1
        adj[y, x] = adj[x, y] = adj[x, y] + 1
    return adj, features

def load_data(dataset):
    """ Load datasets
    :param dataset: name of the input graph dataset
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """
    if dataset == 'google':
         #zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
         adj = nx.adjacency_matrix(nx.read_edgelist("data/google.txt"))
         features = sp.identity(adj.shape[0])
    elif dataset == 'cit-Patents':
        # zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
        adj = nx.adjacency_matrix(nx.read_edgelist("data//cit-Patents.txt"))
        features = sp.identity(adj.shape[0])
    elif dataset == 'er':
        adj_matrix = np.load('data/er1w.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset == 'ba':
        adj_matrix = np.load('data/ba1w.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset == 'ws1w':
        adj_matrix = np.load('data/ws1w.npy')
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg001':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.01.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg002':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.02.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg003':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.03.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg004':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.04.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg005':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.05.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])

    elif dataset =='tg006':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.06.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg007':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.07.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg008':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.08.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg009':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.09.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg010':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.1.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset =='tg011':
        adj_matrix = np.load('toy_graph/toy_graph_adj_0.11.npy')
        G = nx.from_numpy_matrix(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset == 'ws1k':
        adj_matrix = np.load('data/ws1k.npy')
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset == 'ws1b':
        adj_matrix = np.load('data/ws1b.npy')
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset == 'ws3b':
        adj_matrix = np.load('data/ws3b.npy')
        G = nx.from_numpy_array(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset == 'ws3k':
        adj_matrix = np.load('data/ws3k.npy')
        G = nx.from_numpy_array(adj_matrix,create_using=nx.DiGraph)
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])
    elif dataset== 'sj100':
        adj = nx.adjacency_matrix(nx.read_edgelist('data/suijitu100.txt'))
        features = sp.identity(adj.shape[0])
    elif dataset =='wb100':
        adj = nx.adjacency_matrix(nx.read_edgelist('data/wubiaodu100.txt'))
        features = sp.identity(adj.shape[0])
    elif dataset =='enron':
        adj = nx.adjacency_matrix(nx.read_edgelist('data/Email-Enron.txt'))
        features = sp.identity(adj.shape[0])
    elif dataset =='ba1000':
        adj = nx.adjacency_matrix(nx.read_edgelist('data/ba1000.txt'))
        features = sp.identity(adj.shape[0])
    elif dataset == 'twitter':
         #zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
         adj = nx.adjacency_matrix(nx.read_edgelist("data/ego-twitter.txt"))
         features = sp.identity(adj.shape[0])
    elif dataset == 'phonecalls':
         #zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
         adj = nx.adjacency_matrix(nx.read_edgelist("data/phonecalls.txt"))
         features = sp.identity(adj.shape[0])
    elif dataset == 'InternetAS':
         #zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
         adj = nx.adjacency_matrix(nx.read_edgelist("data/InternetAS.txt"))
         features = sp.identity(adj.shape[0])
    elif dataset == 'InternetAS2':
         #zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
         adj = nx.adjacency_matrix(nx.read_edgelist("data/InternetAS2.txt"))
         features = sp.identity(adj.shape[0])
    elif dataset == 'collaboration':
         #zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
         adj = nx.adjacency_matrix(nx.read_edgelist("data/collaboration.txt"))
         features = sp.identity(adj.shape[0])
    elif dataset.startswith('ba3k_'):
        adj = nx.adjacency_matrix(nx.read_edgelist(f'data/ba_test/{dataset}.txt'))
        features = sp.identity(adj.shape[0])  

    elif dataset == 'florida':
        return load_florida()
    elif dataset == 'brainfly':
        adj = nx.adjacency_matrix(nx.read_edgelist("data/bn-fly-drosophila.txt"))
        features = sp.identity(adj.shape[0])
    elif dataset == 'brainhuman': 
        adj = nx.adjacency_matrix(nx.read_edgelist("data/bn-human-BNU.txt"))
        features = sp.identity(adj.shape[0])
    elif dataset == 'brainmouse': 
        adj = nx.adjacency_matrix(nx.read_edgelist("data/bn-mouse.txt"))
        features = sp.identity(adj.shape[0])

    elif dataset == 'enzyme':
        return load_enzyme()
    elif dataset == 'protein':
        return load_protein()
    elif dataset == 'amazon':
         #zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
         adj = nx.adjacency_matrix(nx.read_edgelist("data//Amazon.txt"))
         features = sp.identity(adj.shape[0])
    elif dataset ==  'caph':
        # zf.ZipFile("data//google.txt.zip").extract("google.txt", "data//")
        adj = nx.adjacency_matrix(nx.read_edgelist("data//CAPh.txt"))
        features = sp.identity(adj.shape[0])
    elif dataset == 'cath':
        #9800
        adj = nx.adjacency_matrix(nx.read_edgelist('data//CA-HepTh.txt'))
        features = sp.identity(adj.shape[0])
    elif dataset =='cit':
        adj = nx.adjacency_matrix(nx.read_edgelist("data//Cit-HepPh.txt"))
        features = sp.identity(adj.shape[0])
    elif dataset == 'cmat':
        adj = nx.adjacency_matrix(nx.read_edgelist("data//CA-CondMat.txt"))
        features = sp.identity(adj.shape[0])
    elif dataset == 'p2p':
        adj = nx.adjacency_matrix(nx.read_edgelist("data//p2p-Gnutella08.txt"))
        features = sp.identity(adj.shape[0])

    elif dataset == 'society':
        adj = nx.adjacency_matrix(nx.read_edgelist("data//soc.txt"))
        features = sp.identity(adj.shape[0])

    elif dataset == 'Google':
        adj = nx.adjacency_matrix(nx.read_edgelist("data//Google.txt"))
        features = sp.identity(adj.shape[0])
    elif dataset in ('cora', 'citeseer', 'pubmed'):
        # Load the data: x, tx, allx, graph
        names = ['x', 'tx', 'allx', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data//ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, tx, allx, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data//ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.from_dict_of_lists(graph)
        adj = nx.adjacency_matrix(graph)
    else:
        raise ValueError('Undefined dataset!')
    return adj, features

def load_label(dataset):
    """ Load node-level labels
    :param dataset: name of the input graph dataset
    :return: n-dim array of node labels, used for clustering task
    """
    if dataset == 'google':
        raise ValueError('No ground truth community for Google dataset')
    elif dataset in ('cora', 'citeseer', 'pubmed'):
        names = ['ty', 'ally']
        objects = []
        for i in range(len(names)):
            with open("data\\ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        ty, ally = tuple(objects)
        test_idx_reorder = parse_index_file("data//ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)
        if dataset == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
        labels = sp.vstack((ally, ty)).tolil()
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # One-hot to integers
        labels = np.argmax(labels.toarray(), axis = 1)
    else:
        raise ValueError('Undefined dataset!')
    return labels