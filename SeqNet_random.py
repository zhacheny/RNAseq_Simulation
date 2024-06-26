import numpy as np
import networkx as nx
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

class GeneNetworkGenerator:
    def __init__(self, nmin=10, navg=50, sigma=50, k=4, p=0.1, nu=0.3):
        self.nmin = nmin
        self.navg = navg
        self.sigma = sigma
        self.k = k
        self.p = p
        self.nu = nu
        self.modules = []  # Initialize an empty list to store module states
        self.subgraphs = []  # Initialize an empty list to store subgraphs

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

    def generate_network_structure(self, num_genes, num_modules):
        G = nx.Graph()
        selected_genes = set()
        module_sizes = [self.generate_module_size() for _ in range(num_modules)]
        
        for module_idx in range(num_modules):
            module_size = module_sizes[module_idx]
            if module_idx == 0:
                module = np.random.choice(num_genes, size=(module_size,), replace=False)
            else:
                # sample a link gene
                link_gene = np.random.choice(list(selected_genes))

                # set up probabilities for selected non link genes
                p_unnormalized = []
                for k in range(num_genes):
                    if k == link_gene:
                        p_unnormalized.append(0)
                    elif k in selected_genes:
                        p_unnormalized.append(self.nu)
                    else:
                        p_unnormalized.append(1)
                p = np.array(p_unnormalized) / np.sum(p_unnormalized)

                # sample non link genes
                remaining = np.random.choice(num_genes, p=p, size=(module_size - 1,), replace=False)
                module = [link_gene] + list(remaining)

            # update list of modules and selected genes
            self.modules.append(module)
            selected_genes.update(module)

            # Adjust k to be less than or equal to the number of nodes in the module
            k = min(self.k, len(module) - 1)
            
            # Create a Watts-Strogatz small-world graph for the module
            module_graph = nx.watts_strogatz_graph(len(module), k, self.p)
            # using barabasi albert method
            # module_graph = nx.barabasi_albert_graph(n=len(module), m=1, )

            # Add module state to the modules list
            self.subgraphs.append(module_graph)
        
            # Add edges to the global graph G
            for edge in module_graph.edges():
                G.add_edge(module[edge[0]], module[edge[1]])
        
        
        return G, list(selected_genes)

    def generate_precision_matrices(self):
        Omegas = []
        epsilon = 1e-4
        for i, module in enumerate(self.modules):
            P = len(module)
            W = np.random.uniform(low=-1, high=1, size=(P, P))
            W = np.tril(W) + np.tril(W, -1).T
            subgraph = self.subgraphs[i]
            adj_mat = nx.to_numpy_array(subgraph)
            _Omega = adj_mat * W
            _Omega = _Omega + epsilon * np.eye(*_Omega.shape)

            # correct for PD
            lam = np.real(np.linalg.eigvals(_Omega))
            lam_max, lam_min = lam.max(), lam.min()
            c = lam_max * (10**(-2.5)) - lam_min
            _Omega = _Omega + c * np.eye(_Omega.shape[0])

            # Ensure _Omega is symmetric
            _Omega = (_Omega + _Omega.T) / 2

            Omegas.append(_Omega)
        return Omegas

    def get_reference_rnaseq_data(self, num_genes, num_samples):
        return np.random.rand(num_samples, num_genes)  # Example data

    def gaussian_to_rnaseq(self, gaussian_values, empirical_cdf):
        num_samples, num_genes = gaussian_values.shape
        rnaseq_data = np.zeros((num_samples, num_genes))
        for i in range(num_genes):
            rnaseq_data[:, i] = np.interp(norm.cdf(gaussian_values[:, i]), np.linspace(0, 1, len(empirical_cdf[:, i])), empirical_cdf[:, i])
        return rnaseq_data

    def generate_synthetic_data(self, num_genes, num_samples, num_modules):
        # Generate the network structure
        G, selected_genes = self.generate_network_structure(num_genes, num_modules)

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

        # Generate the precision matrices
        Omegas = self.generate_precision_matrices()

        # Generate local expression vectors
        Xs = []
        for i, module in enumerate(self.modules):
            P = len(module)  # num genes in this module
            Omega = Omegas[i]  # precision matrix
            cov = np.linalg.solve(Omega, np.eye(Omega.shape[0]))  # Omega cov = I
            L = np.linalg.cholesky(cov)  # Cholesky factorization of covariance matrix
            Z = np.random.randn(num_samples, P)
            X = np.einsum("ij, bj -> bi", L, Z)  # zero mean multivariate Gaussian
            Xs.append(X)

        # Global assembly
        X_global = np.zeros((num_samples, num_genes))
        for module, X in zip(self.modules, Xs):
            X_global[:, module] += X

        # Check for unselected genes
        unselected = [g for g in range(num_genes) if g not in selected_genes]
        X_global[:, unselected] = np.random.randn(num_samples, len(unselected))

        # Normalization by gene
        g, c = np.unique(np.sort(np.hstack(self.modules)), return_counts=True)  # c represents the Mis in the paper; number of modules in which gene i appears
        counts_dict = {k: v for k, v in zip(g, c)}
        counts_dict.update({k: 0 for k in unselected})
        norm = np.array([counts_dict[k] if counts_dict[k] != 0 else 1. for k in range(num_genes)])
        norm = 1. / np.sqrt(norm)
        X_global = X_global / norm

        print(X_global)

        # Save the synthetic RNA-seq data to a CSV file
        df = pd.DataFrame(X_global.T, columns=[f'Sample_{i+1}' for i in range(num_samples)])
        df.index = [f'Gene_{i+1}' for i in range(num_genes)]
        df.to_csv('/Users/zhangch/Downloads/synthetic_rnaseq_data.csv')

        print("Synthetic RNA-seq data saved to 'synthetic_rnaseq_data.csv'.")

# Create an instance of the class
generator = GeneNetworkGenerator()

# Generate synthetic data
generator.generate_synthetic_data(1000, 1000, 10)

# Print module states
for idx, module in enumerate(generator.modules):
    print(f"Module {idx+1}:")
    print(f"Genes: {module}")
    print(f"Graph edges: {generator.subgraphs[idx].edges()}")
