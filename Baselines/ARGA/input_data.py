import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)



def load_data(dataset):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    if dataset == 'ws3k':
        # Load custom WS3K adjacency matrix and initialize features
        adj_matrix = np.load('/public/chenjiawen/tst/cndp/data/ws3k.npy')
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])

        # Create placeholders for ws3k-specific outputs
        y_test = np.zeros((adj.shape[0], 10))  # Adjust size if label dimensions differ
        tx = sp.lil_matrix((adj.shape[0], 10))  # Placeholder for test feature matrix
        ty = np.zeros((adj.shape[0], 10))       # Placeholder for test labels
        test_mask = np.zeros(adj.shape[0], dtype=bool)
        labels = np.zeros((adj.shape[0], 10))   # Placeholder for labels

    else:
        for i in range(len(names)):
            with open("/public/chenjiawen/tst_baselines/ARGA-master/ARGA/arga/data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("/public/chenjiawen/tst_baselines/ARGA-master/ARGA/arga/data/ind.{}.test.index".format(dataset))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset == 'citeseer':
            # Fix isolated nodes in Citeseer dataset
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_test, tx, ty, test_mask, np.argmax(labels, axis=1)


def load_alldata(dataset_str):
    """Load data based on dataset name."""
    if dataset_str == 'ws3k':
        # Load ws3k-specific data and create placeholders
        adj_matrix = np.load('data/ws3k.npy')
        G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph())
        adj = nx.adjacency_matrix(G)
        features = sp.identity(adj.shape[0])

        # Placeholders for label matrices and masks
        y_train = np.zeros((adj.shape[0], 10))  # Adjust size as needed
        y_val = np.zeros((adj.shape[0], 10))
        y_test = np.zeros((adj.shape[0], 10))
        train_mask = np.zeros(adj.shape[0], dtype=bool)
        val_mask = np.zeros(adj.shape[0], dtype=bool)
        test_mask = np.zeros(adj.shape[0], dtype=bool)
        labels = np.zeros((adj.shape[0], 10))  # Adjust size if needed

    else:
        # Load data for other datasets like citeseer
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Citeseer-specific fix for isolated nodes
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, np.argmax(labels, axis=1)
