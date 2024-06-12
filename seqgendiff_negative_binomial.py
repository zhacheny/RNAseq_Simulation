import numpy as np
import pandas as pd

def generate_negative_binomial_dataset(input_csv, output_csv, noise_std, hasControl, gene_up, gene_down, gene_no, dispersion):
    # Load real RNA-seq dataset
    real_data = pd.read_csv(input_csv, index_col=0)

    # Apply log transformation to the original data
    log_real_data = np.log1p(real_data)
    # Parameter settings
    np.random.seed(1)
    p, n = log_real_data.shape  # Number of genes and samples

    # Extract mean vector mu and perturbation matrix Omega
    mu = log_real_data.mean(axis=1).values

    # Omega = log_real_data.values - mu.reshape(-1, 1)
    Omega = noise_std * np.random.randn(p, n)

    # Generate design matrix Pi_x
    x = np.random.binomial(1, 0.5, n)  # Randomly generate two groups
    Pi_x = x.reshape(1, -1)  # Convert x to row vector

    # Generate gene effects with adjustments for specific genes
    b = np.zeros(p)  # Initialize b with zeros

    # Assign values to b based on gene effect
    for gene in gene_up:
        if gene in real_data.index:
            b[real_data.index.get_loc(gene)] = 1  # Positive effect for up-regulated genes

    for gene in gene_down:
        if gene in real_data.index:
            b[real_data.index.get_loc(gene)] = -1  # Negative effect for down-regulated genes

    for gene in gene_no:
        if gene in real_data.index:
            b[real_data.index.get_loc(gene)] = 0  # No effect for no-change genes

    # Create new expression matrices Theta and tilde_Theta
    new_Theta = mu.reshape(-1, 1) @ np.ones((1, n)) + Omega
    new_tilde_Theta = mu.reshape(-1, 1) @ np.ones((1, n)) + b.reshape(-1, 1) @ Pi_x + Omega

    # Calculate lambda values
    lambda_Theta = 2 ** new_Theta
    lambda_tilde_Theta = 2 ** new_tilde_Theta

    # Define dispersion parameter (phi)
    phi = dispersion

    # Function to generate Negative Binomial counts
    def generate_negative_binomial(lam, phi):
        r = 1 / phi
        p = r / (r + lam)
        counts = np.random.negative_binomial(r, p)
        return counts

    # Generate Negative Binomial-distributed count data
    counts = generate_negative_binomial(lambda_Theta, phi)
    tilde_counts = generate_negative_binomial(lambda_tilde_Theta, phi)
    
    if hasControl:
        # Create combined dataset
        data_matrix = np.hstack((counts, tilde_counts))

        # Create sample labels
        labels = ['Control'] * n + ['Treatment'] * n

        # Create DataFrame
        genes = real_data.index.tolist()
        samples = [f'Sample_{i+1}' for i in range(n)] + [f'Sample_{i+1+n}' for i in range(n)]
        df = pd.DataFrame(data_matrix.T, index=samples, columns=genes)
        # Add group labels
        df['Group'] = labels

        # Move 'Group' column to the first column
        cols = ['Group'] + [col for col in df.columns if col != 'Group']
        df = df[cols]
    else:
        # Create combined dataset
        data_matrix = tilde_counts

        # Create DataFrame
        genes = real_data.index.tolist()
        samples = [f'Sample_{i+1+n}' for i in range(n)]
        df = pd.DataFrame(data_matrix.T, index=samples, columns=genes)

    df.T.to_csv(output_csv, index=True)

    print(f"The dataset has been generated and saved as '{output_csv}'.")

# Example usage
input_csv = '/Users/zhangch/Downloads/GSE212277_merged.counts.bulk.csv'
output_csv = '/Users/zhangch/Downloads/generated_GSE212277_negative_binomial.csv'
# Define gene effect lists
gene_up = ['Gnai3', 'Narf', 'Cav2']
gene_down = ['Klf6', 'Scmh1', 'Cox5a']
gene_no = ['Fer', 'Xpo6', 'Axin2']
generate_negative_binomial_dataset(input_csv, output_csv, 1, True, gene_up, gene_down, gene_no, dispersion=0.1)
