import numpy as np 

def edge_index_to_adj_mat(edge_index, num_nodes=None):
    if not num_nodes:
        num_nodes = 1 + np.max(edge_index[0])
    adj_mat = np.zeros((num_nodes, num_nodes))
    adj_mat[edge_index[0]][:, edge_index[1]] = 1. 
    return adj_mat 

def adj_mat_to_edge_index(adj_mat):
    return np.vstack(np.where(adj_mat))

def generate_cov_mat_from_adjacency(adj_mat, rho_min=0.7, eps=1e-10):
    """
    Generate a covariance matrix from an adjacency matrix.

    Parameters:
    adj_mat <numpy.ndarray> - symmetric square matrix representing adjacency
    rho_min <float> - minimum value of pairwise correlation to assign to edge. 
                    default = 0.5.
    eps <float> - Lower bound on eigenvalues of covariance matrix. 


    Returns:
    cov <numpy.ndarray> - symmetric positive definite nxn covariance matrix. 
    """

    assert np.all(adj_mat == adj_mat.T), 'Adjacency matrix not symmetric.'
    assert rho_min > 0. and rho_min < 1., "Invalid correlation value. enter a value in the range (0., 1.)"
    # simulate the correlation matrix 
    n = len(adj_mat)
    W = np.random.uniform(low=rho_min, high=1., size=(n, n))
    W = np.triu(W, 1) 
    sign = np.random.choice([-1., 1.,], replace=True, size=(n, n))
    W = W*sign
    W = W + W.T 
    cov = np.eye(*W.shape) + W*adj_mat

    # correct for positive definiteness  
    lam, _ = np.linalg.eig(cov) 
    lam_min = abs(np.sort(lam)[0])
    cov = (eps + lam_min)*np.eye(*cov.shape) + cov  

    return cov