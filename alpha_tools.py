import numpy as np
import networkx as nx
import scipy.linalg as la
from ipyparallel import Client
import matplotlib.pyplot as plt
import pandas as pd
import importlib
import os
import nb_general as nb
import socket
import scipy.sparse as sp
from sklearn.preprocessing import normalize

client = Client()
client.ids
dview = client[:]
dview.map(os.chdir, ['/Users/student/Documents/research/Jones/alpha_centrality']*8)
dview.execute("import numpy as np")
dview.execute("import networkx as nx")
dview.execute("import scipy.linalg as la")
dview.execute("import nb_general as nb")
dview.block = True


def mu_pr(G,mu,eps=.8,S=None,T=None,tau=None,personalization=None):
    """
    Computes mu-pagerank for a G with a given value of mu
    
    Parameters:
        G (networkx Graph)
        mu (float)
        eps (float): damping factor
        S (ndarray): edge to node matrix
        T (ndarray): node to edge matrix
        tau (ndarray): backtracking operator
       
    Returns:
       mu_pagerank (ndarray): mu-pagerank of nodes of G
    """
    # Find necessary matrices
    if S is None or T is None:
        S, T = nb.create_s_t(G)
        
    if tau is None:
        tau = nb.create_tau(G)
        
    if personalization is None:
        Dinv = sp.diags(np.array(1/(T@S).sum(axis=1))[:,0])

        # Get personalization vector
        u = normalize((T.T@Dinv).sum(axis=1),norm='l1',axis=0)
        personalization = dict()
        for i in range(len(u)):
            personalization[i] = u[i]
        
    # Create Hmu
    new_graph = nx.from_scipy_sparse_matrix(S@T-(1-mu)*tau)
    
    # Compute pagerank
    edge_pr = np.array(list(nx.pagerank(new_graph,alpha=eps,personalization=personalization).values()))
    
    # Project to nodes
    return T@edge_pr

def mu_matrix(G,eps=.8,begin=0,end=100,divisions=20,personalization=None):
    """
    Calculate mu-pagerank for various mu
    
    Parameters:
        G (networkx Graph)
        eps (float): damping factor
        begin (float): first value of mu
        end (float): last value of mu
        divisions (int): number of values of mu to compute
        
    Return:
        A (ndarray): matrix showing mu-pagerank values where rows represent nodes and columns represent mu
    """
    # Get mu values
    mus = np.linspace(begin,end,divisions)
    S, T = nb.create_s_t(G)
    tau = nb.create_tau(G)
    
    # Get params
    param_G = [G for i in range(len(mus))]
    param_eps = [eps for i in range(len(mus))]
    param_S = [S for i in range(len(mus))]
    param_T = [T for i in range(len(mus))]
    param_tau = [tau for i in range(len(mus))]
    param_personalization = [personalization for i in range(len(mus))]
    # Compute mu-pagerank
    output = list(dview.map(mu_pr,param_G,mus,param_eps,param_S,param_T,param_tau,param_personalization))
    n = len(G.nodes())
    output = [arr.reshape((n,1)) for arr in output]
    
    # Make matrix
    A = np.hstack(tuple(output))

    return A

def make_graph(G=None,A=None,eps=.8,begin=0,end=100,divisions=20,personalization=None,legend=True):
    """
    Create graph of mu-pagerank
    
    Parameters:
        G (networkx graph)
        eps (float): damping factor
        A (ndarray): mu-matrix
        begin (float): first value of mu
        end (float): last value of mu
        divisions (int): number of values of mu to compute
    """
    # Get A if not provided
    if A is None:
        A = mu_matrix(G,eps,begin,end,divisions,personalization)
    # Make colors if number of nodes is less than 20
    if A.shape[0] <= 20:
        cm = plt.cm.get_cmap('tab20')
        domain = np.linspace(begin,end,divisions)
        for i in range(A.shape[0]):
            plt.plot(domain,A[i,:],'-o',color=cm.colors[i],label=i)
        if legend:
            plt.legend(bbox_to_anchor=(1.05,1))
        plt.title(f"$\mu$-PageRank")
        plt.xlabel("mu")
        plt.ylabel("PageRank")
        plt.tight_layout()
        plt.show()
    
    else:
        domain = np.linspace(begin,end,divisions)
        for i in range(A.shape[0]):
            plt.plot(domain,A[i,:],'-o',label=i)
        if legend:
            plt.legend(bbox_to_anchor=(1.05,1))
        plt.title(f"$\mu$-PageRank")
        plt.xlabel("mu")
        plt.ylabel("PageRank")
#         plt.tight_layout()
        plt.show()