
import numpy as np
import networkx as nx

def fit_power_law(subgraph):
    degrees = [d for n, d in subgraph.degree()]
    
    degree_counts = np.bincount(degrees)
    degree_counts = degree_counts[1:]  # 去掉度为0的情况
    x = np.arange(1, len(degree_counts) + 1)
    y = degree_counts
    mask = y != 0
    x = x[mask]
    y = y[mask] 
    x_log = np.log(x)
    y_log = np.log(y)
    coeff = np.polyfit(x_log, y_log, 1) 
    y_coeff = -coeff[0]
    return y_coeff





def avg_shortest_path_length(G):
    try:
        # 检查图是否是连通的无向图
        if not G.is_directed():
            if nx.is_connected(G):
                average_shortest_path_length = nx.average_shortest_path_length(G)
            else:
                # 图不连通，计算每个连通组件的平均最短路径长度
                connected_components = list(nx.connected_components(G))
                if len(connected_components) > 1:
                    # print('Input graph is not connected; computing for each component:')
                    avg_spl_per_component = []
                    for component in connected_components:
                        subgraph = G.subgraph(component)
                        if nx.is_connected(subgraph):
                            avg_spl_per_component.append(nx.average_shortest_path_length(subgraph))
                    if avg_spl_per_component:
                        average_shortest_path_length = sum(avg_spl_per_component) / len(avg_spl_per_component)
                    else:
                        average_shortest_path_length = None
                else:
                    print("Graph is not connected.")
                    average_shortest_path_length = None
        else:
            raise ValueError("The input graph should be an undirected graph.")
    except nx.NetworkXError:
        average_shortest_path_length = None

    return  average_shortest_path_length 



def subgraph_property(G):

    density = nx.density(G)
    sparsity = 1 - density

    average_degree = sum(dict(G.degree()).values()) / len(G)


    average_clustering_coefficient = nx.average_clustering(G)

    # 平均最短路径长度

    average_shortest_path_length = avg_shortest_path_length(G)
    # 生成对应的随机网络 (Erdős–Rényi 模型)
    n = G.number_of_nodes()
    m = G.number_of_edges()
    G_ER = nx.gnm_random_graph(n, m)

    # 计算随机网络的平均聚类系数和平均最短路径长度
    C_ER = nx.average_clustering(G_ER)
    try:
        L_ER = avg_shortest_path_length(G_ER)
    except nx.NetworkXError:
        L_ER = None

    # 计算 r_C 和 r_L
    if average_clustering_coefficient != 0 and C_ER is not None:
        r_C = abs(average_clustering_coefficient - C_ER) / average_clustering_coefficient
    else:
        r_C = None

    if average_shortest_path_length is not None and L_ER is not None and average_shortest_path_length != 0:
        r_L = abs(average_shortest_path_length - L_ER) / average_shortest_path_length
    else:
        r_L = None

 
    return r_L,r_C, sparsity, average_degree, average_clustering_coefficient, average_shortest_path_length

