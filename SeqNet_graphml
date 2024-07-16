import numpy as np, os
import pandas as pd
import networkx as nx
import os

import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric as tg

import matplotlib as mpl
from matplotlib import pyplot as plt

# load graph
input_path = "/projects/compsci/karuturi/common-data-dir/ALZ/human/latest_graphml/"
fpath = input_path + "./bdFiltered_APP_Metabolism_undirected_filt_node_edge.graphml"
g = nx.read_graphml(fpath)
g = g.to_undirected()
nx.draw(g);

gene_list = [g._node[node]['name'] for node in g.nodes()]
node_id_to_gene_dict = {i:g for i, g in enumerate(gene_list)}
gene_to_node_id_dict = {g:i for i, g in node_id_to_gene_dict.items()}
A = nx.to_numpy_array(g)
print(f"Num genes = {len(A)}")
edge_index = np.vstack(list(np.where(A)))
node_degrees = dict(g.degree())

gene_degrees = {}
for gene, node_id in zip(gene_list, node_degrees):
    gene_degrees[gene] = node_degrees[node_id]
gene_degrees = {k: v for k, v in sorted(gene_degrees.items(), key=lambda item: item[1], reverse=True)}

def get_k_hop_subset(hub_gene, edge_index, num_hops=2, ):
    """
    hub_gene <str> : 'Name of gene around which to extract subgraph.
    edge_index <numpy.ndarray> : 2xnumedges matrix of edge connections.
    num_hops <int> : Max. path length.
    """
    node_idx = gene_to_node_id_dict[hub_gene]
    subset, edge_index, mapping, edge_mask = tg.utils.k_hop_subgraph(node_idx=node_idx, num_hops=num_hops, edge_index=torch.tensor(edge_index).to(torch.int64), num_nodes=len(gene_list))
    subset = list(subset.detach().cpu().numpy())
    subset_genes = [node_id_to_gene_dict[idx] for idx in subset]
    return subset_genes


subgraph_gene_list = get_k_hop_subset('APP', edge_index)
print(subgraph_gene_list)

subg_node_ids = [gene_to_node_id_dict[gene] for gene in subgraph_gene_list]
subg_adjmat = A[subg_node_ids][:, subg_node_ids]
g = nx.from_numpy_array(subg_adjmat)
g_new = nx.barabasi_albert_graph(len(g), m=2)

plt.figure()
nx.draw(g)
plt.savefig("g.png")  
plt.close() 
plt.figure()
nx.draw(g_new)
plt.savefig("g_new.png")
plt.close() 

original_adjmat = A
subgraph_node_ids = subg_node_ids
new_subgraph_adjmat = nx.to_numpy_array(g_new)

new_adjmat = A.copy()  # Make a copy to avoid modifying the original
new_adjmat[np.ix_(subgraph_node_ids, subgraph_node_ids)] = new_subgraph_adjmat
new_edge_index = np.vstack(list(np.where(new_adjmat)))

def generate_precision_matrix(adjmat, epsilon=1e-5, k=2.5):
    P = len(adjmat)
    W = np.random.uniform(low=-1, high=1, size=(P, P))
    W = np.tril(W) + np.tril(W, -1).T
    Omega_tilde = adjmat * W
    Omega_tilde = Omega_tilde + epsilon * np.eye(*Omega_tilde.shape)

    # Ensure Omega is positive definite
    lambda_max = np.max(np.real(np.linalg.eigvals(Omega_tilde)))
    lambda_min = np.min(np.real(np.linalg.eigvals(Omega_tilde)))
    c = lambda_max * (10**(-k)) - lambda_min
    Omega = Omega_tilde + c * np.eye(*Omega_tilde.shape)
    
    return Omega

def generate_data(adjmat, epsilon=1e-4, num_samples=100):
    Omega = generate_precision_matrix(adjmat)
    cov = np.linalg.inv(Omega)  # Use np.linalg.inv to get the covariance matrix
    L = np.linalg.cholesky(cov)  # Cholesky factorization of covariance matrix
    Z = np.random.randn(num_samples, Omega.shape[0])
    X = np.einsum("ij, bj -> bi", L, Z)  # Zero mean multivariate Gaussian
    return X

X1 = generate_data(original_adjmat)
X2 = generate_data(new_adjmat)

# Add gene names and sample IDs, then save to CSV
def save_to_csv(data, filename, gene_list, num_samples):
    df = pd.DataFrame(data.T, columns=[f'Sample_{i+1}' for i in range(num_samples)])
    df.index = gene_list
    df.to_csv(filename)
    print(f"Data saved to '{filename}'.")

script_directory = os.path.dirname(os.path.abspath(__file__))
save_to_csv(X1, script_directory + '/original_data.csv', gene_list, X1.shape[0])
save_to_csv(X2, script_directory + '/perturbed_data.csv', gene_list, X2.shape[0])
