import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import networkx as nx
from networkx.algorithms.connectivity import edge_connectivity, average_node_connectivity
import scipy.sparse as sp

def create_k(G):
    # Get necessary matrices
    A = np.array(nx.adjacency_matrix(G).todense())
    D = np.diag(list(dict(G.degree()).values()))
    I = np.eye(D.shape[0])
    return np.block([[A,D-I],[-I,np.zeros((D.shape[0],D.shape[0]))]])

def create_s_t(G):
    dg = G.to_directed()
    new_nodes = []
    for e in G.edges():
        new_nodes.append(e)
        new_nodes.append(e[::-1])
    # Find S and T
    m = len(G.edges())
    n = len(G.nodes())
    s_cols = np.array([np.where(np.array(list(dg.nodes()))==e[0])[0][0] for e in new_nodes])
    s_rows = np.array([i for i in range(2*m)])
    t_cols = np.array([i for i in range(2*m)])
    t_rows = np.array([np.where(np.array(list(dg.nodes()))==e[1])[0][0] for e in new_nodes])
    data = np.ones(len(new_nodes))
    S = sp.coo_matrix((data,(s_rows,s_cols)))
    T = sp.coo_matrix((data,(t_rows,t_cols)))
    return S, T

def create_tau(G):
    m = len(G.edges())
    sp_mini_tau = sp.coo_matrix([[0,1],[1,0]])
    mat = np.array([[None for i in range(m)] for j in range(m)])
    for i in range(m):
        mat[i,i] = sp_mini_tau
    return sp.bmat(mat)


def to_edge_space(G, B=False, graph=True, ret_tau = False, S=None, T=None, tau=None):
    direct = G.to_directed()
    # Find S and T
    if S is None or T is None:
        S, T = create_s_t(G)
        
    # Create tau
    if tau is None:
        tau = create_tau(G)
    
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

def rank(x):
    """
    Computes standardize ranking of vector which accounts for ties
    
    Parameters:
        x (ndarray)
    
    Returns:
        rank (ndarray)
    """
    # Create ranking array and sorted array
    copy = np.copy(x)
    sorted_x = np.sort(copy)
    rank = np.zeros_like(copy)
    # Check rank of each node
    for i, val in enumerate(copy):
        # Account for ties by using np.min
        rank[i] = np.min(np.where(np.around(sorted_x,decimals=12)==round(val,12))[0])
        
    return rank

class NBEigVals:
    """
    Class with tools to investigate formula above
    
    Attributes:
        G (networkx Graph)
        n (int) - number of nodes
        m (int) - number of edges
        A (ndarray) - adjacency matrix of G
        D (ndarray) - degree matrix
        I (ndarray) - identity the size of D
        K (ndarray) - k matrix
        y_matrix (ndarray) - matrix of "y"igenvectors
        x_matrix (ndarray) - matrix of eigenvectors of A
        kvecs (ndarray) - eigenvectors of K
        mu_vals (ndarray) - eigenvalues of K
        lambda_vals (ndarray) - eigenvalues of K
        conv_matrix (ndarray) - matrix C such that x_matrix@C=y_matrix
        S (ndarray) - vertex to edge matrix
        T (ndarray) - edge to vertex matrix
        C (ndarray) - edge space adjacency matrix
        B (ndarray) - nonbacktracking matrix
    """
    
    def __init__(self, G):
        """
        Initialize NBEigVals object
        
        Parameters:
            G (networkx Graph)
        """
        self.G = G
        self.n = len(self.G.nodes())
        self.A, self.D, self.I, self.K = self.get_matrices()
        self.mu_vals, self.kvecs = la.eig(self.K)
        self.y_matrix = self.kvecs[self.K.shape[0]//2:,:]
#         self.y_matrix = self.y_matrix/la.norm(self.y_matrix,axis=0)
        for i in range(self.y_matrix.shape[1]):
            self.y_matrix[:,i] = self.y_matrix[:,i]/np.sqrt(self.y_matrix[:,i]@self.y_matrix[:,i])
        self.lambda_vals, self.x_matrix = la.eig(self.A)
        self.conv_matrix = self.x_matrix.T@self.y_matrix
        self.S, self.T = create_s_t(self.G)
        self.C, self.B, self.tau = to_edge_space(G,B=True,graph=False,ret_tau=True)
        self.m = self.B.shape[0]//2
        
    
    def calc_y(self,idx):
        """
        Get the y portion of eigenvectors of K.

        Parameters:
            K (2n,2n) ndarray
            idx (int): idx of eigenvector

        Return:
            y (n,) ndarray
        """
        # Get eigenvalues and eigenvectors
        return self.mu_vals[idx], self.y_matrix[:,idx]

    def get_matrices(self):
        """
        Get necessary matrices for formula

        Parameters:
            G (networkx graph)

        Return:
            A (n,n) ndarray: adjacency matrix
            D (n,n) ndarray: degree matrix
            I (n,n) ndarray: identity matrix
            K (2n,2n) ndarray: K matrix
        """
        A = nx.adjacency_matrix(self.G).todense()
        D = np.diag(np.sum(np.array(A),axis=1))
        I = np.eye(D.shape[0])
        K = create_k(self.G)
        return A, D, I, K

    def formula(self,idx_x,idx_y):
        """
        Evaluate formula.

        Parameters:
            idx_x (int): index of lambda
            idx_y (int): index of mu

        Return:
            mu_pos (float): evaluation of formula (negative)
            mu_neg (float): evaluation of formula (positive)
        """
        # Get needed parameters
        lambda_ = self.lambda_vals[idx_x]
        x = self.x_matrix[:,idx_x]
        y = self.y_matrix[:,idx_y]
        
        # Calculate positive formula
        pos_num = lambda_*x.T@y+np.sqrt((lambda_*x.T@y)**2-4*x.T@y*x.T@(self.D-self.I)@y)
        pos_den = 2*x.T@y
        mu_pos = pos_num/pos_den

        # Calculate negative formula
        neg_num = lambda_*x.T@y-np.sqrt((lambda_*x.T@y)**2-4*x.T@y*x.T@(self.D-self.I)@y)
        neg_den = 2*x.T@y
        mu_neg = neg_num/neg_den

        return mu_pos, mu_neg
    
    def y_formula(self, idx_y):
        """
        Evaluate formula created with only y.
        
        Parameters:
            idx_y (int): index of mu
            
        Return:
            mu_pos (float): evaluation of formula (positive)
            mu_neg (float): evaluation of formula (negative)
        """
        # Get need parameters
        y = self.y_matrix[:,idx_y]
        
        # Calculate positive formula
        mu_pos = (y.T@self.A@y+np.sqrt((y.T@self.A@y)**2-4*y.T@(self.D-self.I)@y))/2
        
        # Calculate negative formula
        mu_neg = (y.T@self.A@y-np.sqrt((y.T@self.A@y)**2-4*y.T@(self.D-self.I)@y))/2
        
        return mu_pos, mu_neg

    def compare_mu(self, idx, tol, y_formula):
        """
        Compare a given eigenvalue of K with the formula
        evaluate with every choice of lambda.
        Creates print-out.

        Parameters:
            idx (int): index of eigenvalue
            tol (float): tolerance level
            y_formula (bool): determine f to use y_formula
        """
        # Check if idx is valid
        if idx >= 2*self.n:
            raise ValueError("idx too large")

        # Find mu and y
        mu, y = self.calc_y(idx)

        print(f"mu: {mu}")
        
        if y_formula:
            cur_pos, cur_neg = self.y_formula(idx)
            if np.abs(mu - cur_pos) < tol:
                pos = True
            else:
                pos = False
                print(f"\tIncorrect: {cur_pos}")
            if np.abs(mu - cur_neg) < tol:
                neg = True
            else:
                neg = False
                print(f"\tIncorrect: {cur_neg}")
            print(f"\tPositive: {pos}\tNegative: {neg}")
            
                
        else:
            for i in range(self.n):
                cur_pos, cur_neg = self.formula(i,idx)

                if np.abs(mu - cur_pos) < tol:
                    pos = True
                else:
                    pos = False
                if np.abs(mu - cur_neg) < tol:
                    neg = True
                else:
                    neg = False
                if neg == False and pos == False:
                    print(f"\t{i} - Positive: {pos}\tNegative: {neg}\tx^Ty={self.x_matrix[:,i]@y}")
                else:
                    print(f"\t{i} - Positive: {pos}\tNegative: {neg}\tx^Ty={self.x_matrix[:,i]@y}")


    def check_all_vals(self, tol=1e-8, y_formula=False):
        """
        Check formula for all values of mu.
        Creates print-out.

        Parameters:
            tol (float): tolerance level
            y_formula (bool): determine whether to use y formula
        """
        for i in range(2*self.n):
                print(f"\n\t\t--------------------{i+1}---------------------\n")
                self.compare_mu(i, tol, y_formula)
    
    def linear_combinations(self):
        """
        Solve Ax=b where A=X and y is a column of Y
        """
        for i in range(2*self.n):
            print(la.solve(self.x_matrix,self.y_matrix[:,i]))
            
    def orthogonality(self,x_idx=None,y_idx=None):
        """
        Look at patterns of x^Ty
        """
        if x_idx is not None:
            if y_idx is not None:
                print(self.x_matrix[:,x_idx]@self.y_matrix[:,y_idx])
            else:
                for i in range(2*self.n):
                    print(self.x_matrix[:,x_idx]@self.y_matrix[:,i],self.x_matrix[:,x_idx]@self.D@self.y_matrix[:,i])
        if y_idx is not None:
            if x_idx is None:
                print(self.x_matrix@self.y_matrix[:,y_idx])
        if x_idx is None:
            if y_idx is None:
                for i in range(2*self.n):
                    print(self.x_matrix.T@self.y_matrix[:,i])
     
    def deformed_laplacian(self, idx):
        """
        Return deformed laplacian
        """
        return self.mu_vals[idx]*self.A-self.D