import numpy as np
import networkx as nx
import scipy.linalg as la
from ipyparallel import Client
import matplotlib.pyplot as plt
import pandas as pd

client = Client()
client.ids
dview = client[:]
dview.execute("import numpy as np")
dview.execute("import networkx as nx")
dview.execute("import scipy.linalg as la")
dview.block = True

def mu_pr(G,mu,eps=.8):
    def create_s_t(G):
        direct = G.to_directed()
        # Find S and T
        S = np.zeros((len(direct.edges),len(G.nodes)))
        T = np.zeros((len(G.nodes),len(direct.edges)))
        for i,a in enumerate(direct.edges):
            for j,b in enumerate(G.nodes):
    #             print(a,b)
                if a[1] == b:
                    S[i,j] = 1
                if a[0] == b:
                    T[j,i] = 1
        return S, T


    def to_edge_space(G, B=False, graph=True, ret_tau = False):
        direct = G.to_directed()
        # Find S and T
        S = np.zeros((len(direct.edges),len(G.nodes)))
        T = np.zeros((len(G.nodes),len(direct.edges)))
        for i,a in enumerate(direct.edges):
            for j,b in enumerate(G.nodes):
    #             print(a,b)
                if a[1] == b:
                    S[i,j] = 1
    #                 print('S Here')
                if a[0] == b:
    #                 print('T Here')
                    T[j,i] = 1
        # Create tau
            tau = np.zeros((len(direct.edges),len(direct.edges)))
            for i,a in enumerate(direct.edges):
                for j,b in enumerate(direct.edges):
                    if a[0]==b[1] and a[1]==b[0]:
                        tau[i][j] = 1
        # Create edge matrix
        if B:
            if graph:
                if ret_tau:
                    return nx.Graph(S@T), nx.Graph(S@T-tau), nx.Graph(tau)
                return nx.Graph(S@T), nx.Graph(S@T-tau)
            if ret_tau:
                return S@T, S@T - tau, tau
            return S@T, S@T - tau
        if graph:
            if ret_tau:
                return nx.Graph(S@T), nx.Graph(tau)
            return nx.Graph(S@T)
        if ret_tau:
            return  S@T, tau
        return S@T
    S, T = create_s_t(G)
    C, tau = to_edge_space(G,graph=False,ret_tau=True)
    Dinv = np.diag(1/(T@S).sum(axis=1))
    u = T.T@Dinv@np.ones(len(G.nodes()))
    u = u/la.norm(u,ord=1)
    personalization = dict()
    for i in range(len(u)):
        personalization[i] = u[i]
    new_graph = nx.from_numpy_array(C-(1-mu)*tau)
    edge_pr = np.array(list(nx.pagerank(new_graph,alpha=eps,personalization=personalization).values()))
    return T@edge_pr

def make_graph(A,_):
    cm = plt.cm.get_cmap('tab20')
    domain = np.linspace(0,100,20)
    for i in range(A.shape[1]):
        plt.plot(domain,A[:,i],'-o',color=cm.colors[i],label=i)
    plt.legend(bbox_to_anchor=(1.05,1))
    plt.title(f"$\mu$-PageRank")
    plt.xlabel("mu")
    plt.ylabel("PageRank")
    plt.tight_layout()
    plt.savefig(f"example_2.pdf")
    plt.close()

mus = np.linspace(0,100,20)
evaluations = np.zeros((20,2000))

# Create random geometric graphs
for _ in range(1):
    G = list(nx.connected_component_subgraphs(nx.random_geometric_graph(20,.4)))[0]
    print(_)
    nx.write_gml(G,'example_2_graph.gml')
    eps = 0
    while eps == 0:
        eps = np.random.random()
    output = dview.map(mu_pr,[G for i in range(len(mus))],mus,[.8 for i in range(len(mus))])
    A = np.vstack(output)
    # evaluation = np.sign(A-np.roll(A,1,axis=0))[1:,:].sum(axis=0)
    # evaluations[:,_] = evaluation
    make_graph(A,_)
    # print(f"{_} BROKE")
    # continue

# Create gnp graph
for _ in range(0):
    try:
        print(int(1000+_))
        G = list(nx.connected_component_subgraphs(nx.gnp_random_graph(20,.2)))[0]
        eps = 0
        while eps == 0:
            eps = np.random.random()
        output = dview.map(mu_pr,[G for i in range(len(mus))],mus)
        A = np.vstack(output)
        evaluation = np.sign(A-np.roll(A,1,axis=0))[1:,:].sum(axis=0)
        evaluations[:,int(1000+_)] = evaluation
        make_graph(A,int(1000+_))
    except:
        print(f"{int(1000+_)} BROKE")
        continue 

evaluations = np.array(evaluations)
data = pd.DataFrame(evaluations)
data.to_csv("monotonicity_check.csv")


