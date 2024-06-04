import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sim_polyester_data(csv_file_path, gene_up, gene_down, gene_no, output_file_path, seed=42):
    """
    Simulate gene expression data using the Polyester model.
    
    Parameters:
    - csv_file_path: Path to the CSV file containing the gene expression parameters.
    - gene_up: List of genes to be up-regulated.
    - gene_down: List of genes to be down-regulated.
    - gene_no: List of genes with no regulation.
    - output_file_path: Path to save the simulated data.
    - seed: Random seed for reproducibility.
    
    Returns:
    - DataFrame containing the simulated gene expression data.
    """
    # Set the random seed
    np.random.seed(seed)

    # Read CSV file
    params_df = pd.read_csv(csv_file_path, index_col=0)
    all_genes = params_df.index.to_list()

    # Calculate the mean expression value (μ) and dispersion parameter (r) for each gene
    mu = params_df.iloc[:, 1:].mean(axis=1)
    r = mu / 3  # Assume the dispersion parameter is 1/3 of the mean value

    # Ensure all r values are greater than zero
    r = r.apply(lambda x: max(x, 1e-3))

    # Define the parameters for the negative binomial distribution
    size = r.values
    prob = size / (size + mu.values)

    # Define fold change λ
    fold_changes = np.ones(len(mu))  # Initialize to 1 (no regulate)

    # Set fold change for gene_up to be up-regulated, e.g., λ > 1
    fold_changes[params_df.index.isin(gene_up)] = np.random.randint(2, 5, size=params_df.index.isin(gene_up).sum())

    # Set fold change for gene_down to be down-regulated, e.g., λ < 1
    fold_changes[params_df.index.isin(gene_down)] = 1 / np.random.randint(2, 5, size=params_df.index.isin(gene_down).sum())

    # Adjust μ according to fold change
    mu_adj = mu * fold_changes

    # Update probability
    prob_adj = size / (size + mu_adj.values)

    # Simulate gene expression data
    num_genes = len(size)
    num_samples = 12
    expression_data = np.random.negative_binomial(size[:, np.newaxis], prob_adj[:, np.newaxis], size=(num_genes, num_samples))

    # Convert the result to DataFrame
    gene_ids = params_df.index
    sample_ids = [f'Sample_{i}' for i in range(1, num_samples + 1)]

    df = pd.DataFrame(expression_data, index=gene_ids, columns=sample_ids)

    # Save the DataFrame to a CSV file
    df.to_csv(output_file_path)

    return df

# demo
csv_file_path = '/Users/zhangch/Downloads/GSE212277_merged.counts.bulk.csv'
gene_up = ['Gnai3', 'Klf6', 'Scmh1']
gene_down = ['Pbsn', 'H19', 'Scml2']
gene_no = ['Narf', 'Cav2', 'Cox5a']
output_file_path = '/Users/zhangch/Downloads/simulated_expression_data.csv'

simulated_data = sim_polyester_data(csv_file_path, gene_up, gene_down, gene_no, output_file_path)
print(simulated_data.head())
