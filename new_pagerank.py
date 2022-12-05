import networkx as nx
import numpy as np
import scipy.linalg as la
import nb_general as nb
import random
import os
import scipy.sparse as sp

def pagerank(G,v,alpha=0,var="limit",v_type="personalized"):
    """
    Get PageRank distribution for given node, alpha, variation

    Parameter:
        G (networkx Graph)
        v (node index)
        alpha (float): jumping factor
        var (string): PageRank variation ("limit","reg", or "nb")
        v_type (string): Variation on v ("personalized" or "uniform")

    Returns:
        pr_vector (ndarray)
    """
    # Get nodes
    nodes = np.array(list(G.nodes()))
    n = len(nodes)

    # Get v
    if v_type == "personalized":
        # Get personalization vector
        vector_v = np.zeros(n)
        vector_v[v] = 1
    elif v_type == "uniform":
        vector_v = np.ones(n)/n
    else:
        raise TypeError("Invalid v_type")

    # Get matrices
    # Dinv = np.diag(1/np.array(list(dict(G.degree())).values()))
    Dinv = np.diag(1/np.array(list(dict(G.degree()).values())))
    A = nx.adjacency_matrix(G).todense()

    # Get damping factor
    eps = 1 - alpha

    # Calculate pagerank based on variation
    if var == "limit":
        # Calculate limit PageRank vector
        return 1/(1+eps)*vector_v.T+eps/(1+eps)*vector_v.T@Dinv@A

    if var == "reg":
        # Convert personalization vector to dictionary for networkx
        dictionary_v = {nodes[i]: v for i, v in enumerate(nodes)}

        # Calculate Pagerank
        pr_vector = nx.pagerank(G,eps,personalization=dictionary_v)
        return np.array(list(dict(pr_vector).values()))

    if var == "nb":
        # Convert personalization vector to dictionary for networkx
        dictionary_v = {nodes[i]: v for i, v in enumerate(nodes)}

        # Calculate nb pagerank
        S, T = nb.create_s_t(G)
        tau = nb.create_tau(G)
        B = S@T-tau
        nb_graph = nx.from_scipy_sparse_matrix(B)

        # Calculate pagerank
        nb_pr_vector = nx.pagerank(nb_graph,eps,personalization=dictionary_v)

        # Project to nodes
        nb_pr_vector = np.array(list(dict(nb_pr_vector).values()))
        T = T.todense()
        projection = T@nb_pr_vector
        return projection

def distribution_array(G,alpha=0,var="limit",v_type="personalized"):
    """
    Create array of distributions based on PageRank vectors

    G (networkx)
    var (str) - type of variation
    v_type (str) - type of variation

    Returns:
    distribution_array
    """
    nodes = np.array(list(G.nodes()))
    n = len(nodes)
    pr_vectors = np.array([pagerank(G,v,alpha,var,v_type) for v in range(n)])

    return np.vstack(pr_vectors)

def pr_dist(G,u_idx,v_idx,distributions):
    """
    Calculate pagerank distance between two vertices

    G (networkx)
    u_idx (node index)
    v_idx (node index)
    distributions (ndarray)

    Returns:
        pr_dist (float): PageRank distance
    """

    # Get square root inverse of D
    Dinv2 = np.diag(1/np.sqrt(np.array(list(dict(G.degree()).values()))))

    # Get nodes
    nodes = np.array(list(G.nodes()))

    # Calculate dist
    return np.linalg.norm(distributions[u_idx,:]@Dinv2-distributions[v_idx,:]@Dinv2)

def dist(G,p,q,distributions=None,alpha=0,var="limit",v_type="personalized"):
    """
    Calculate PageRank distance between two distributions

    Parameters:
    G (networkx Graph)
    p (ndarray): probability distribution
    q (ndarray): probability distribution
    distributions (ndarray): array of PageRank vectors
    alpha (float): jumping factor
    var (str): type of variation
    v_type (str): type variation type

    Return:
    dist (float): PageRank distance
    """
    # Get list of nodes
    nodes = np.array(list(G.nodes()))
    n = len(nodes)

    # Get all possible u and v
    param_u = [u for u in range(n) for i in range(n)]
    param_v = [v for i in range(n) for v in range(n)]

    if distributions is None:
        distributions = distribution_array(G,alpha,var,v_type)

    # Compute distances
    distances = np.array([pr_dist(G,u_idx,v_idx,distributions) for u_idx in range(n) for v_idx in range(n)])

    p_vals = p[param_u]
    q_vals = q[param_v]

    return (p_vals*q_vals*distances).sum()

def pr_cluster(G,k,alpha=0,tol=1e-8,var="limit",v_type="personalized",distributions = None,pr_dist=None):
    """
    Get clusters

    Parameters:
    G (networkx)
    k (int): number of clusters
    alpha (float): jumping factor
    tol (float): convergence tolerance
    var (str)
    v_type (str)
    pr_dist (ndarray): pagerank distances

    Returns:
    node_labels (list)
    """
    nodes = np.array(list(G.nodes()))
    n = len(nodes)
    pi = np.array(list(dict(G.degree()).values()))/nx.volume(G,nodes)

    # Choose random centers
    random_centers = np.random.choice(range(n),size=k,p=pi,replace=False)

    if distributions is None:
        distributions = distribution_array(G,alpha,var,v_type)

    C = np.array(distributions[random_centers,:])

    error = np.inf

    while error > tol:
        node_labels = {}

        # Save old centers
        old_C = np.copy(C)

        # Find best cluster for each node
        for i, v in enumerate(nodes):
            node_distribution = distributions[i,:]

            # Initialize distances = np.zeros(K)
            distances = np.zeros(k)

            # Check distance of each cluster
            for j in range(k):
                distances[j] = dist(G,node_distribution,C[j,:],distributions,alpha,var,v_type)
            
            # Find nearest cluster
            node_labels[i] = np.argmin(distances)

        # Get new centers
        for i in range(k):
            node_clusters = [j for j, v in node_labels.items() if v==i]
            try:
                C[i,:] = distributions[[j for j, v in node_labels.items() if v==i],:].mean(axis=0)
            except:
                raise ZeroDivisionError("Empty Cluster. Try Less Clusters.")
        error = np.linalg.norm(C-old_C)
    
    for i in list(node_labels.keys()):
        node_labels[nodes[i]] = node_labels.pop(i)
    return node_labels

