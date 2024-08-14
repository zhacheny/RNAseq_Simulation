import pandas as pd
import networkx as nx
import numpy as np, os
import torch
import torch_geometric as tg

def get_k_hop_subset(hub_gene, edge_index, num_hops=2, gene_to_node_id_dict=None, node_id_to_gene_dict=None):
    node_idx = gene_to_node_id_dict[hub_gene]
    subset, edge_index, mapping, edge_mask = tg.utils.k_hop_subgraph(node_idx=node_idx, num_hops=num_hops, edge_index=torch.tensor(edge_index).to(torch.int64), num_nodes=len(gene_to_node_id_dict))
    subset = list(subset.detach().cpu().numpy())
    subset_genes = [node_id_to_gene_dict[idx] for idx in subset]
    return subset_genes

def gen_A_perturbed_gene(g, gene_list, edge_index, gene_to_node_id_dict, node_id_to_gene_dict):
    # Subgraph extraction and modification
    A = nx.to_numpy_array(g)
    print(f"Num genes = {len(A)}")
    subgraph_gene_list = get_k_hop_subset('APP', edge_index, gene_to_node_id_dict=gene_to_node_id_dict, node_id_to_gene_dict=node_id_to_gene_dict)
    print(subgraph_gene_list)

    subg_node_ids = [gene_to_node_id_dict[gene] for gene in subgraph_gene_list]
    subg_adjmat = A[subg_node_ids][:, subg_node_ids]
    print(f"Num genes for sub graph= {len(subg_adjmat)}")
    g = nx.from_numpy_array(subg_adjmat)
    g_new = nx.barabasi_albert_graph(len(g), m=2)

    # Create the new adjacency matrix with perturbed subgraph
    new_adjmat = A.copy()
    new_subgraph_adjmat = nx.to_numpy_array(g_new)
    new_adjmat[np.ix_(subg_node_ids, subg_node_ids)] = new_subgraph_adjmat

    return new_adjmat


def generate_cov_mat_from_adjacency(adj_mat, rho_min=0.5, eps=1e-10):
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


def generate_poisson_dataset(input_csv, output_csv, noise_std, hasControl, graphml_path, number_sample, r):
    # Load real RNA-seq dataset
    real_data = pd.read_csv(input_csv, index_col=0)
    real_data.index = real_data.index.str.upper()
    
    # Parameter settings
    np.random.seed(1)
    p, n = real_data.shape  # Number of genes and samples

    # Load graph and generate adjacency matrix
    g = nx.read_graphml(graphml_path)
    g = g.to_undirected()

    gene_list = [g.nodes[node]['name'] for node in g.nodes()]
    
    # Find intersection of genes in RNA-seq data and graph
    common_genes = list(set(gene_list) & set(real_data.index))
    print(f"Number of common genes: {len(common_genes)}")
    
    # Filter RNA-seq data and mu vector to only include common genes
    filtered_data = real_data.loc[common_genes]
    filtered_mu = np.log2(filtered_data.mean(axis=1).values)  # log mean

    # Update p to the number of common genes
    p = len(common_genes)
    
    # Generate filtered adjacency matrix A for common genes
    filtered_indices = [gene_list.index(g) for g in common_genes]
    A = nx.to_numpy_array(g)[filtered_indices, :][:, filtered_indices]
    edge_index = np.vstack(list(np.where(A)))

    # Create mapping dictionaries
    gene_to_node_id_dict = {g: i for i, g in enumerate(gene_list)}
    node_id_to_gene_dict = {i: g for g, i in gene_to_node_id_dict.items()}

    # Generate perturbed adjacency matrix
    A_perturbed = gen_A_perturbed_gene(g, gene_list, edge_index, gene_to_node_id_dict, node_id_to_gene_dict)
    A_perturbed = A_perturbed[filtered_indices, :][:, filtered_indices]

    # Use the perturbed adjacency matrix to generate covariance matrix
    cov = generate_cov_mat_from_adjacency(A)
    cov_perturbed = generate_cov_mat_from_adjacency(A_perturbed)

    mean = filtered_mu

    # Generate Omega with the new covariance matrix for all samples
    Theta = np.random.multivariate_normal(mean, cov, 1)[0]
    Theta_perturbed = np.random.multivariate_normal(mean, cov_perturbed, 1)[0]
    print(Theta.shape)
    
    # Function to generate Negative Binomial counts with exception handling
    def safe_negative_binomial(lam, r):
        p = r / (r + lam)
        try:
            return np.random.negative_binomial(r, p)
        except ValueError:
            return np.zeros_like(lam, dtype=int)


    # Generate Negative Binomial-distributed count data with exception handling
    counts = np.array([safe_negative_binomial(2**Theta, r) for _ in range(number_sample)])
    print(counts.shape)
    
    if hasControl:
        # Create combined dataset
        tilde_counts = np.array([safe_negative_binomial(2**Theta_perturbed, r) for _ in range(number_sample)])
        data_matrix = np.hstack((counts.T, tilde_counts.T))

        # Create sample labels
        labels = [0] * number_sample + [1] * number_sample

        # Create DataFrame
        genes = real_data.index.tolist()
        samples = [f'Sample_{i+1}' for i in range(number_sample)] + [f'Sample_{i+1+number_sample}' for i in range(number_sample)]
        print(len(samples), data_matrix.shape)
        df = pd.DataFrame(data_matrix.T, index=samples, columns=common_genes)
        # Add group labels
        df['Label'] = labels

        # Move 'Label' column to the first column
        cols = [col for col in df.columns if col != 'Label'] + ["Label"]
        df = df[cols]
    else:
        # Create combined dataset
        data_matrix = counts

        # Create DataFrame
        samples = [f'Sample_{i+1+number_sample}' for i in range(number_sample)]
        df = pd.DataFrame(data_matrix, index=samples, columns=common_genes)

    df.to_csv(output_csv, index=True)

    print(f"The dataset has been generated and saved as '{output_csv}'.")

# Example usage
script_directory = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_directory, 'data/GSE212277_merged.counts.bulk.csv')
output_csv = os.path.join(script_directory, 'output/out_new_NBD_GSE212277_merged.counts.bulk.csv')
graphml_path = '/projects/compsci/karuturi/common-data-dir/ALZ/human/latest_graphml/bdFiltered_APP_Metabolism_undirected_filt_node_edge.graphml'

gene_up = []
gene_down = []
gene_no = []

generate_poisson_dataset(input_csv, output_csv, 1, True, graphml_path, 500, 10)
