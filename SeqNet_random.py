import numpy as np
import networkx as nx
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

class GeneNetworkGenerator:
    def __init__(self, nmin=10, navg=50, sigma=50):
        self.nmin = nmin
        self.navg = navg
        self.sigma = sigma

    def generate_module_size(self):
        size = self.nmin + np.random.negative_binomial(self.navg - self.nmin, self.sigma / (self.sigma + self.navg - self.nmin))
        return min(size, 200)  # prevent generate extremely large module size

    def generate_symmetric_weight_matrix(self, size):
        W = np.random.uniform(-1, 1, (size, size))
        W = (W + W.T) / 2  # Ensure symmetry
        return W

    def ensure_positive_definite(self, matrix):
        min_eig = np.min(np.real(np.linalg.eigvals(matrix)))
        if min_eig < 0:
            matrix -= 10 * min_eig * np.eye(matrix.shape[0])
        return matrix

    def generate_network_structure(self, num_genes):
        G = nx.Graph()
        gene_list = list(range(num_genes))
        selected_genes = set()
        
        while len(selected_genes) < num_genes:
            module_size = self.generate_module_size()
            module_genes = np.random.choice(gene_list, size=min(module_size, num_genes - len(selected_genes)), replace=False)
            selected_genes.update(module_genes)

            for i in range(len(module_genes)):
                for j in range(i + 1, len(module_genes)):  # Avoid self-loops by starting j from i + 1
                    G.add_edge(module_genes[i], module_genes[j])
        
        # Ensure all selected genes are added to the graph
        for gene in selected_genes:
            if gene not in G:
                G.add_node(gene)
        
        # Ensure there are no isolated nodes
        isolated_nodes = list(nx.isolates(G))
        for isolated_node in isolated_nodes:
            # Connect isolated node to a random existing node
            random_gene = np.random.choice(list(G.nodes))
            G.add_edge(isolated_node, random_gene)
        
        return G, list(selected_genes)

    def generate_precision_matrix(self, G):
        adjacency_matrix = nx.adjacency_matrix(G).todense()
        print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
        weight_matrix = self.generate_symmetric_weight_matrix(adjacency_matrix.shape[0])
        weight_matrix = np.multiply(weight_matrix, adjacency_matrix)
        weight_matrix = self.ensure_positive_definite(weight_matrix)
        return weight_matrix

    def get_reference_rnaseq_data(self, num_genes):
        return np.random.rand(1000, num_genes)  # Example data

    def gaussian_to_rnaseq(self, gaussian_values, empirical_cdf):
        num_samples, num_genes = gaussian_values.shape
        rnaseq_data = np.zeros((num_samples, num_genes))
        for i in range(num_genes):
            rnaseq_data[:, i] = np.interp(norm.cdf(gaussian_values[:, i]), np.linspace(0, 1, len(empirical_cdf[:, i])), empirical_cdf[:, i])
        return rnaseq_data

    def generate_synthetic_data(self, num_genes, num_samples, path):
        # Generate the network structure
        G, selected_genes = self.generate_network_structure(num_genes)

        # Print the structure of the graph
        print("Graph information:")
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")

        # Draw the network
        plt.figure(figsize=(6, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', edge_color='gray')
        plt.title("Generated Gene Network")
        plt.show()

        # Generate the precision matrix
        precision_matrix = self.generate_precision_matrix(G)

        print("Precision Matrix Shape:", precision_matrix.shape)
        # print(precision_matrix)

        # Example reference RNA-seq data
        reference_data = self.get_reference_rnaseq_data(num_genes)
        empirical_cdf = np.percentile(reference_data, np.linspace(0, 100, num_genes), axis=0)

        # Generate multivariate normal expression values
        mean_vector = np.zeros(num_genes)
        cov_matrix = np.linalg.solve(precision_matrix, np.eye(precision_matrix.shape[0]))  # Compute covariance matrix

        # Ensure mean_vector and cov_matrix have matching dimensions
        assert mean_vector.shape[0] == cov_matrix.shape[0], "mean_vector and cov_matrix dimensions do not match"

        gaussian_values = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)

        synthetic_rnaseq_data = self.gaussian_to_rnaseq(gaussian_values, empirical_cdf)

        # Print the generated RNA-seq data
        print("Synthetic RNA-seq Data:")
        print(synthetic_rnaseq_data)

        # Save the synthetic RNA-seq data to a CSV file
        df = pd.DataFrame(synthetic_rnaseq_data.T, columns=[f'Sample_{i+1}' for i in range(num_samples)])
        df.index = [f'Gene_{i+1}' for i in range(num_genes)]
        df.to_csv(path + 'synthetic_rnaseq_data.csv')

        print("Synthetic RNA-seq data saved to 'synthetic_rnaseq_data.csv'.")

# Create an instance of the class
generator = GeneNetworkGenerator()

# initialization parameters
num_genes = 1000
num_samples = 100
path = '/Users/zhangch/Downloads/'
# Generate synthetic data
generator.generate_synthetic_data(num_genes, num_samples, path)
