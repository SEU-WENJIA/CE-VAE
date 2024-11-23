import networkx as nx
import numpy as np
import pandas as pd
# Define the data to visualize
import pickle
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import scipy.io as io
import zipfile as zf
from input_data import load_data




def gravity(G,centrality, r):
    '''
    Paper: Identifying influential speaders by gravity models 2019   nature scientificreports
    Contribute: The paper proposed a novel method to caculate the mix-centrality by gravity theory
    - G： the networks of Snapshot graph
    - centrality: the type centrality of graph(generally list)
    - r: the radius of the neighbourhood of  centrality caculation to node v
    '''
    grav = {}
    for node in (G.nodes()):
        grav[node] = 0
        neighbour_nodes = list(G.neighbors(node))
        for neighbour in  neighbour_nodes:
            if (nx.shortest_path_length(G,source = node , target  = neighbour))>r:
                break
            if (node not in grav):
                grav[node] = 0
            if  (neighbour == node):
                break
            grav[node] += (centrality[neighbour] * centrality[neighbour])/((nx.shortest_path_length(G,source = node , target  = neighbour))**2)
            if ((nx.shortest_path_length(G,source = node , target  = neighbour)))<r:
                for n in G.neighbors(neighbour):
                    if (n not in neighbour_nodes):
                        neighbour_nodes.append(n)
        neighbour_nodes = []
    return grav




import pickle


# datasets = ['Kongsfjorden', 'Weddell_Sea',  'Caribbean_Reef', 'FloridaIslandE1','FloridaIslandE3', 'Lough_Hyne']
datasets = ['ba1w_16']
for dataset in datasets:
    adj, _ = load_data(dataset)

    G = nx.from_scipy_sparse_array(adj,create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G))
    r = 2
    degree = nx.degree_centrality(G)
    with open("centrality/%s_degree.pkl"%dataset, "wb") as f:
        pickle.dump(degree, f)

    core =  nx.core_number(G)
    with open("centrality/%s_core.pkl"%dataset, "wb") as f:
        pickle.dump(core, f)

    pagerank = nx.pagerank(G)
    with open("centrality/%s_pagerank.pkl"%dataset, "wb") as f:
        pickle.dump(pagerank, f)

    eigenvector = nx.eigenvector_centrality(G)
    with open("centrality/%s_eigenvector.pkl"%dataset, "wb") as f:
        pickle.dump(eigenvector, f)
    print(f' dataset {dataset} done!')

    
    betweenness = nx.betweenness_centrality(G)
    with open("centrality/%s_betweenness.pkl"%dataset, "wb") as f:
        pickle.dump(betweenness, f)

    
    gravitydegree = gravity(G,degree,r)
    with open("centrality/%s_gravitydegree.pkl"%dataset, "wb") as f:
        pickle.dump(gravitydegree, f)

    gravitycore = gravity(G, core, r)
    with open("centrality/%s_gravitycore.pkl"%dataset, "wb") as f:
        pickle.dump(gravitycore, f)

    gravitypagerank = gravity(G,pagerank,r)
    with open("centrality/%s_gravitypagerank.pkl"%dataset, "wb") as f:
        pickle.dump(gravitypagerank , f)



    gravitybetweenness = gravity(G, betweenness, r)
    with open("centrality/%s_gravitybetweenness.pkl"%dataset, "wb") as f:
        pickle.dump(gravitybetweenness, f)


    gravityeigenvector = gravity(G, eigenvector, r)
    with open("centrality/%s_gravityeigenvector.pkl"%dataset, "wb") as f:
        pickle.dump(gravityeigenvector, f)


    closeness = nx.closeness_centrality(G)
    with open("centrality/%s_closeness.pkl"%dataset, "wb") as f:
        pickle.dump(closeness, f)

    gravitycloseness = gravity(G, closeness, r)
    with open("centrality/%s_gravitycloseness.pkl"%dataset, "wb") as f:
        pickle.dump(gravitycloseness, f)

    print(f' dataset {dataset} done!')


# datasets = []
# for k in range(11, 31, 1):
#     for p in range(6, 9, 1):
#         filename = f'ws3k_{k}_{p /10.0:.1f}'
#         datasets.append(filename)

# for dataset in datasets:
#     adj, _ = load_data(dataset)

#     G = nx.from_scipy_sparse_array(adj)
#     G.remove_edges_from(nx.selfloop_edges(G))
#     r = 2
#     degree = nx.degree_centrality(G)
#     with open("centrality/%s_degree.pkl"%dataset, "wb") as f:
#         pickle.dump(degree, f)

#     core =  nx.core_number(G)
#     with open("centrality/%s_core.pkl"%dataset, "wb") as f:
#         pickle.dump(core, f)

#     pagerank = nx.pagerank(G)
#     with open("centrality/%s_pagerank.pkl"%dataset, "wb") as f:
#         pickle.dump(pagerank, f)

#     print(f' dataset{dataset} done!')

    # betweenness = nx.betweenness_centrality(G)
    # with open("centrality/%s_betweenness.pkl"%dataset, "wb") as f:
    #     pickle.dump(betweenness, f)


    # closeness = nx.closeness_centrality(G)
    # with open("centrality/%s_closeness.pkl"%dataset, "wb") as f:
    #     pickle.dump(closeness, f)


    # print(f' dataset{dataset} done!')
    # gravitydegree = gravity(G,degree,r)
    # with open("centrality/%s_gravitydegree.pkl"%dataset, "wb") as f:
    #     pickle.dump(gravitydegree, f)

    # gravitycore = gravity(G, core, r)
    # with open("centrality/%s_gravitycore.pkl"%dataset, "wb") as f:
    #     pickle.dump(gravitycore, f)

    # gravitypagerank = gravity(G,pagerank,r)
    # with open("centrality/%s_gravitypagerank.pkl"%dataset, "wb") as f:
    #     pickle.dump(gravitypagerank , f)


    # print(f' dataset{dataset} done!')

    # gravitybetweenness = gravity(G, betweenness, r)
    # with open("centrality/%s_gravitybetweenness.pkl"%dataset, "wb") as f:
    #     pickle.dump(gravitybetweenness, f)
    # print(f' dataset{dataset} done!')




    # gravitycloseness = gravity(G, closeness, r)
    # with open("centrality/%s_gravitycloseness.pkl"%dataset, "wb") as f:
    #     pickle.dump(gravitycloseness, f)

    # print(f' dataset{dataset} done!')



# datasets = []
# for p in range(1, 3, 1):
#     for k in range(1000, 2000, 50):
#         filename = f'ws3k_{k}_{p /10.0:.1f}'
#         datasets.append(filename)

# for dataset in datasets:
#     adj, _ = load_data(dataset)

#     G = nx.from_scipy_sparse_array(adj)
#     G.remove_edges_from(nx.selfloop_edges(G))
#     r = 2
#     degree = nx.degree_centrality(G)
#     with open("centrality/%s_degree.pkl"%dataset, "wb") as f:
#         pickle.dump(degree, f)

#     core =  nx.core_number(G)
#     with open("centrality/%s_core.pkl"%dataset, "wb") as f:
#         pickle.dump(core, f)

#     pagerank = nx.pagerank(G)
#     with open("centrality/%s_pagerank.pkl"%dataset, "wb") as f:
#         pickle.dump(pagerank, f)

#     print(f' dataset{dataset} done!')



# datasets = []
# for edge in range(30,32,2):

#     filename = f'ba3k_{edge}'
#     datasets.append(filename)

# # datasets = ['amazon','twitter','protein','brainhuman','florida','phonecalls']
# for dataset in datasets:
#     adj, _ = load_data(dataset)

#     G = nx.from_scipy_sparse_array(adj)
#     G.remove_edges_from(nx.selfloop_edges(G))
#     r = 2
#     degree = nx.degree_centrality(G)
#     with open("centrality/%s_degree.pkl"%dataset, "wb") as f:
#         pickle.dump(degree, f)

#     core =  nx.core_number(G)
#     with open("centrality/%s_core.pkl"%dataset, "wb") as f:
#         pickle.dump(core, f)

#     pagerank = nx.pagerank(G)
#     with open("centrality/%s_pagerank.pkl"%dataset, "wb") as f:
#         pickle.dump(pagerank, f)

#     print(f' dataset{dataset} done!')


# # for dataset in datasets:
# #     adj, _ = load_data(dataset)

# #     G = nx.from_scipy_sparse_array(adj)
# #     G.remove_edges_from(nx.selfloop_edges(G))
# #     r = 2
# #     degree = nx.degree_centrality(G)
# #     with open("centrality/%s_degree.pkl"%dataset, "wb") as f:
# #         pickle.dump(degree, f)

# #     core =  nx.core_number(G)
# #     with open("centrality/%s_core.pkl"%dataset, "wb") as f:
# #         pickle.dump(core, f)

# #     pagerank = nx.pagerank(G)
# #     with open("centrality/%s_pagerank.pkl"%dataset, "wb") as f:
# #         pickle.dump(pagerank, f)

# #     print(f' dataset{dataset} done!')

# #     betweenness = nx.betweenness_centrality(G)
# #     with open("centrality/%s_betweenness.pkl"%dataset, "wb") as f:
# #         pickle.dump(betweenness, f)


# #     closeness = nx.closeness_centrality(G)
# #     with open("centrality/%s_closeness.pkl"%dataset, "wb") as f:
# #         pickle.dump(closeness, f)

# #     eigenvector = nx.eigenvector_centrality(G)
# #     with open("centrality/%s_eigenvector.pkl"%dataset, "wb") as f:
# #         pickle.dump(eigenvector, f)


# #     print(f' dataset{dataset} done!')
# #     gravitydegree = gravity(G,degree,r)
# #     with open("centrality/%s_gravitydegree.pkl"%dataset, "wb") as f:
# #         pickle.dump(gravitydegree, f)

# #     gravitycore = gravity(G, core, r)
# #     with open("centrality/%s_gravitycore.pkl"%dataset, "wb") as f:
# #         pickle.dump(gravitycore, f)

# #     gravitypagerank = gravity(G,pagerank,r)
# #     with open("centrality/%s_gravitypagerank.pkl"%dataset, "wb") as f:
# #         pickle.dump(gravitypagerank , f)


# #     print(f' dataset{dataset} done!')

# #     gravitybetweenness = gravity(G, betweenness, r)
# #     with open("centrality/%s_gravitybetweenness.pkl"%dataset, "wb") as f:
# #         pickle.dump(gravitybetweenness, f)
# #     print(f' dataset{dataset} done!')

# #     gravityeigenvector = gravity(G, eigenvector, r)
# #     with open("centrality/%s_gravityeigenvector.pkl"%dataset, "wb") as f:
# #         pickle.dump(gravityeigenvector, f)


# #     gravitycloseness = gravity(G, closeness, r)
# #     with open("centrality/%s_gravitycloseness.pkl"%dataset, "wb") as f:
# #         pickle.dump(gravitycloseness, f)

# #     print(f' dataset{dataset} done!')





# # datasets = []
# # for p in range(3, 9, 1):
# #     for k in range(21, 31, 1):
# #         filename = f'ws3k_{k}_{p /10.0:.1f}'
# #         datasets.append(filename)

# # for dataset in datasets:
# #     adj, _ = load_data(dataset)

# #     G = nx.from_scipy_sparse_array(adj)
# #     G.remove_edges_from(nx.selfloop_edges(G))
# #     r = 2
# #     degree = nx.degree_centrality(G)
# #     with open("centrality/%s_degree.pkl"%dataset, "wb") as f:
# #         pickle.dump(degree, f)

# #     core =  nx.core_number(G)
# #     with open("centrality/%s_core.pkl"%dataset, "wb") as f:
# #         pickle.dump(core, f)

# #     pagerank = nx.pagerank(G)
# #     with open("centrality/%s_pagerank.pkl"%dataset, "wb") as f:
# #         pickle.dump(pagerank, f)

# #     print(f' dataset{dataset} done!')


# # for dataset in datasets:
# #     adj, _ = load_data(dataset)

# #     G = nx.from_scipy_sparse_array(adj)
# #     G.remove_edges_from(nx.selfloop_edges(G))
# #     r = 2
# #     degree = nx.degree_centrality(G)
# #     # with open("centrality/%s_degree.pkl"%dataset, "wb") as f:
# #     #     pickle.dump(degree, f)

# #     core =  nx.core_number(G)
# #     # with open("centrality/%s_core.pkl"%dataset, "wb") as f:
# #     #     pickle.dump(core, f)

#     pagerank = nx.pagerank(G)
#     # with open("centrality/%s_pagerank.pkl"%dataset, "wb") as f:
#     #     pickle.dump(pagerank, f)

#     print(f' dataset{dataset} done!')

#     betweenness = nx.betweenness_centrality(G)
#     with open("centrality/%s_betweenness.pkl"%dataset, "wb") as f:
#         pickle.dump(betweenness, f)


#     closeness = nx.closeness_centrality(G)
#     with open("centrality/%s_closeness.pkl"%dataset, "wb") as f:
#         pickle.dump(closeness, f)

#     eigenvector = nx.eigenvector_centrality(G)
#     with open("centrality/%s_eigenvector.pkl"%dataset, "wb") as f:
#         pickle.dump(eigenvector, f)


#     print(f' dataset{dataset} done!')
#     gravitydegree = gravity(G,degree,r)
#     with open("centrality/%s_gravitydegree.pkl"%dataset, "wb") as f:
#         pickle.dump(gravitydegree, f)

#     gravitycore = gravity(G, core, r)
#     with open("centrality/%s_gravitycore.pkl"%dataset, "wb") as f:
#         pickle.dump(gravitycore, f)

#     gravitypagerank = gravity(G,pagerank,r)
#     with open("centrality/%s_gravitypagerank.pkl"%dataset, "wb") as f:
#         pickle.dump(gravitypagerank , f)


#     print(f' dataset{dataset} done!')

#     gravitybetweenness = gravity(G, betweenness, r)
#     with open("centrality/%s_gravitybetweenness.pkl"%dataset, "wb") as f:
#         pickle.dump(gravitybetweenness, f)
#     print(f' dataset{dataset} done!')

#     gravityeigenvector = gravity(G, eigenvector, r)
#     with open("centrality/%s_gravityeigenvector.pkl"%dataset, "wb") as f:
#         pickle.dump(gravityeigenvector, f)


#     gravitycloseness = gravity(G, closeness, r)
#     with open("centrality/%s_gravitycloseness.pkl"%dataset, "wb") as f:
#         pickle.dump(gravitycloseness, f)

#     print(f' dataset{dataset} done!')