import pandas as pd
import networkx as nx
import numpy as np, os
import networkx as nx

def generate_poisson_dataset(input_csv, output_csv, noise_std, hasControl, gene_up, gene_down, gene_n, graphml_path):
    # Load real RNA-seq dataset
    real_data = pd.read_csv(input_csv, index_col=0)
    real_data.index = real_data.index.str.upper()
    # Apply log transformation to the original data
    log_real_data = np.log1p(real_data)

    # Parameter settings
    np.random.seed(1)
    p, n = log_real_data.shape  # Number of genes and samples

    # Extract mean vector mu
    mu = log_real_data.mean(axis=1).values

    # Load graph and generate adjacency matrix
    g = nx.read_graphml(graphml_path)
    g = g.to_undirected()

    gene_list = [g.nodes[node]['name'] for node in g.nodes()]
    
    # Find intersection of genes in RNA-seq data and graph
    common_genes = list(set(gene_list) & set(log_real_data.index))
    print(f"Number of common genes: {len(common_genes)}")
    
    # Filter RNA-seq data and mu vector to only include common genes
    filtered_data = log_real_data.loc[common_genes]
    filtered_mu = filtered_data.mean(axis=1).values

    # Update p to the number of common genes
    p = len(common_genes)
    
    # Generate filtered adjacency matrix A for common genes
    filtered_indices = [gene_list.index(g) for g in common_genes]
    A_filtered = nx.to_numpy_array(g)[filtered_indices, :][:, filtered_indices]

    # Compute new covariance matrix
    Sigma_new = (A_filtered + noise_std) ** 2

    # Check if Sigma_new is positive semi-definite
    if np.all(np.linalg.eigvals(Sigma_new) >= 0):
        print("The new covariance matrix is positive semi-definite.")
    else:
        print("Warning: The new covariance matrix is not positive semi-definite.")
    import pdb; pdb.set_trace()
    # Generate Omega with the new covariance matrix
    Omega = np.random.multivariate_normal(np.zeros(p), Sigma_new, n).T

    # Generate design matrix Pi_x
    x = np.random.binomial(1, 0.5, n)  # Randomly generate two groups
    Pi_x = x.reshape(1, -1)  # Convert x to row vector

    # Generate gene effects with adjustments for specific genes
    b = np.zeros(p)  # Initialize b with zeros

    # Assign values to b based on gene effect
    for gene in gene_up:
        if gene in filtered_data.index:
            b[filtered_data.index.get_loc(gene)] = 1  # Positive effect for up-regulated genes

    for gene in gene_down:
        if gene in filtered_data.index:
            b[filtered_data.index.get_loc(gene)] = -1  # Negative effect for down-regulated genes

    for gene in gene_n:
        if gene in filtered_data.index:
            b[filtered_data.index.get_loc(gene)] = 0  # No effect for no-change genes

    # Create new expression matrices Theta and tilde_Theta
    new_Theta = filtered_mu.reshape(-1, 1) @ np.ones((1, n)) + Omega
    new_tilde_Theta = filtered_mu.reshape(-1, 1) @ np.ones((1, n)) + b.reshape(-1, 1) @ Pi_x + Omega

    # Function to generate Poisson counts with exception handling
    def safe_poisson(lam):
        try:
            return np.random.poisson(lam=lam)
        except ValueError:
            return np.zeros_like(lam, dtype=int)

    # Generate Poisson-distributed count data with exception handling
    counts = safe_poisson(2 ** new_Theta)
    tilde_counts = safe_poisson(2 ** new_tilde_Theta)
    if hasControl:
        # Create combined dataset
        data_matrix = np.hstack((counts, tilde_counts))

        # Create sample labels
        labels = ['Control'] * n + ['Treatment'] * n

        # Create DataFrame
        samples = [f'Sample_{i+1}' for i in range(n)] + [f'Sample_{i+1+n}' for i in range(n)]
        df = pd.DataFrame(data_matrix.T, index=samples, columns=common_genes)
        # Add group labels
        df['Group'] = labels

        # Move 'Group' column to the first column
        cols = ['Group'] + [col for col in df.columns if col != 'Group']
        df = df[cols]
    else:
        # Create combined dataset
        data_matrix = tilde_counts

        # Create DataFrame
        samples = [f'Sample_{i+1+n}' for i in range(n)]
        df = pd.DataFrame(data_matrix.T, index=samples, columns=common_genes)

    df.T.to_csv(output_csv, index=True)


    print(f"The dataset has been generated and saved as '{output_csv}'.")

# Example usage
script_directory = os.path.dirname(os.path.abspath(__file__))
input_csv = os.path.join(script_directory, 'data/GSE212277_merged.counts.bulk.csv')
output_csv = os.path.join(script_directory, 'output/out_GSE212277_merged.counts.bulk.csv')
graphml_path = '/projects/compsci/karuturi/common-data-dir/ALZ/human/latest_graphml/bdFiltered_APP_Metabolism_undirected_filt_node_edge.graphml'
# Define gene effect lists
# gene_up = ['Gnai3', 'Narf', 'Cav2']
# gene_down = ['Klf6', 'Scmh1', 'Cox5a']
# gene_no = ['Fer', 'Xpo6', 'Axin2']

gene_up = []
gene_down = []
gene_no = []

generate_poisson_dataset(input_csv, output_csv, 1, True, gene_up, gene_down, gene_no, graphml_path)
