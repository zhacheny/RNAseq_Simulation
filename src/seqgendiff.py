import numpy as np
from scipy import stats
import networkx as nx
from .utils import generate_cov_mat_from_adjacency

class SeqGenDiff:
    """
    Generate RNASeq data by setting up heirarchical model.

    Reference:
    ---------
    Gerard, D. (2020). Data-based RNA-seq simulations by binomial thinning. 
    Bmc Bioinformatics, 21, 1-14.
    """

    def _variance(self, mu):
        r = mu / self.disp_factor
        return mu + ((mu**2) / r)
    
    def __init__(
                self, 
                data, 
                data_model='poisson', 
                cov=None,
                adj_mat=None,
                edge_index=None,
                disp_factor=3,):
        """
        data <numpy.ndarray> - numpy matrix of read data - rows are samples columns are genes. 
        data_model <str> - The data generating probability model - Poisson or negative
                            binomial. If negative binomial, an optional dispersion factor
                            may be supplied. Default - poisson. 
                            The parameterization of the mean of the Poisson or negative
                            binomial distribution is 2^theta where theta \sim N (mu, cov) 
                            is a multivariate Gaussian whose mean captures information
                            from real data and the covariance matrix captures pairwise
                            relationship between genes. 
        cov <numpy.ndarray> - (optional) A symmetric p.d. covariance matrix 
                            which captures the intended pairwise relational structure 
                            between genes. By default this will be set to the 
                            identity matrix (i.e. counts for all genes are sampled fully independently). 
        
        adj_mat <numpy.ndarray> - (optional) provide adjacency matrix for gene network. if provided,
                                    used to generate covariance matrix and assigned preferentially
                                    over cov.  
        edge_index <numpy.ndarray> - (optional) provide edge index matrix for gene network. 
                                    if provided, used preferentially over adj_mat and used to generate
                                    covariance matrix. Shape of the edge_index matrix has
                                    to be (2, num edges). Graph is forced to be undirected. 
        disp_factor <float> - The factor by which to scale the mean to obtain 
                              dispersion factor of the negative binomial distribution
                            for sampling. dispersion factor r = mean / disp_factor
                            default: disp_factor = 3. 
        """
        if cov != None:
            assert np.all(cov == cov.T)
        assert data_model.strip().lower() in ['poisson', 'negative_binomial']
        self.disp_factor = disp_factor
        self.means = data.mean(0)
        self.adj_mat = adj_mat 
        self.edge_index = edge_index
        self.cov = cov 
        self.data_model = data_model.strip().lower()
        self.data_dist = stats.poisson if self.data_model == 'poisson' else stats.nbinom
    
    def _get_covariance_matrix(self,):
        if self.edge_index is None and self.adj_mat is None: 
            if self.cov is None:
                return np.eye(len(self.means))
            else:
                return self.cov
        
        if self.edge_index != None:
            adj_mat = np.zeros((len(self.means), len(self.means)))
            adj_mat[edge_index[0]][:, edge_index[1]] = 1. 
            if not np.all(adj_mat == adj_mat.T):
                adj_mat = np.triu(adj_mat, 1)
                adj_mat = adj_mat + adj_mat.T 
            return generate_cov_mat_from_adjacency(adj_mat)
                
        else:
            adj_mat = self.adj_mat
            if not np.all(adj_mat == adj_mat.T):
                adj_mat = np.triu(adj_mat, 1)
                adj_mat = adj_mat + adj_mat.T 
            return generate_cov_mat_from_adjacency(adj_mat)

    def generate_by_network_perturbation(self, n1, n2, k_hop=1, hub_idx=None, m=2):
        """
        Note: This method is only applicable if self.cov is not None or an identity 
        matrix i.e., there is a non trivial network topology over the set of genes. 
        In case of identity covariance, only n1 control samples are generated and returned. 

        Parameters:
        ==========
        n1 <int> - Number of control samples.
        n2 <int> - Number of case samples.
        k_hop <int> - Radius of subgraph to extract around hub gene. By default
                        the k_hop length is set to 1 i.e. the subgraph to perturb 
                        is the hub gene and all the genes directly sharing an edge 
                        with the hub gene. 
        hub_idx <int> - The index of the gene to use as the hub gene. If None is provided, 
                        the node with the highest degree is used. 
        m <int> - Hyperparameter of Barabasi Albert random graph generator for generating 
                    perturbed network. 
        
        
        Returns:
        --------
        X <numpy.ndarray> - Data matrix of size (n1+n2, num genes). Each row 
                            is synthetic read count data for either a case or control
                            sample. 

        """
        # generate n1 samples in normal condition 
        theta_m = np.log2(1e-10 + self.means)
        theta_cov = self._get_covariance_matrix()
        theta = np.random.multivariate_normal(mean=theta_m, cov=theta_cov)
        mu = 2**theta
        if self.data_model == 'poisson':
            args = (mu,)
        else:
            var = self._variance(mu)
            args = ((mu*mu) / (var - mu), mu/var)
        data_dist = self.data_dist(*args)
        x1 = [data_dist.rvs() for _ in range(n1)]
        x1 = np.array(x1)
        
        # check if there is a non trivial covariance matrix and if not return x1
        if np.allclose(theta_cov, np.eye(len(theta_cov))):
            return x1
        
        # generate case samples 
        # rewire the adjacency matrix 
        # set up adjacency matrix 
        if self.edge_index != None:
            adj_mat = np.zeros((len(theta), len(theta)))
            adj_mat[self.edge_index[0]][:, self.edge_index[1]] = 1.
        else:
            adj_mat = self.adj_mat 
        if not np.all(adj_mat == adj_mat.T):
            adj_mat = np.triu(adj_mat, 1)
            adj_mat = adj_mat + adj_mat.T 
        
        # set up the hub node index.
        if hub_idx == None:
            node_degrees = np.sum(adj_mat, axis=1)
            hub_idx = np.argmax(node_degrees)
        
        # get k hop neighborhood 
        k_hop_nodes = np.where(np.linalg.matrix_power(adj_mat, k_hop)[hub_idx])[0]
        k_hop_nodes = list(k_hop_nodes)
        
        # generate a new subgraph for this k hop neighborhood 
        g_sub = nx.barabasi_albert_graph(len(k_hop_nodes), m=m)
        g_sub = g_sub.to_undirected()
        adj_mat_sub = nx.to_numpy_array(g_sub)
        adj_mat[k_hop_nodes][:, k_hop_nodes] = adj_mat_sub

        # generate covariance matrix for this new adjacency matrix 
        theta_cov = generate_cov_mat_from_adjacency(adj_mat)

        # generate the data for the test condition 
        theta = np.random.multivariate_normal(mean=theta_m, cov=theta_cov)
        mu = 2**theta
        if self.data_model == 'poisson':
            args = (mu,)
        else:
            var = self._variance(mu)
            args = ((mu*mu) / (var - mu), mu/var)
        data_dist = self.data_dist(*args)
        x2 = [data_dist.rvs() for _ in range(n2)]
        x2 = np.array(x2)

        # stack, shuffle and return 
        x = np.vstack([x1, x2])
        perm = np.random.permutation(len(x))
        x = x[perm] 
        return x 

    def generate_by_effect_size(self, n1, n2, 
                                down_idx=[], up_idx=[], 
                                fold_change_min=2, fold_change_max=5):
        """
        Parameters:
        ==========

        n1 <int> - Number of control samples.
        n2 <int> - Number of case samples.
        down_idx <list> - Index of genes to be down regulated.
        up_idx <list> - index of genes to be up regulated.
        fold_change_min <float> - Lower bound on fold change.
        fold_change_max <float> - upper bound on fold change. 

        Returns:
        -------
        X <numpy.ndarray> - Data matrix of size (n1+n2, num genes). Each row 
                            is synthetic read count data for either a case or control
                            sample. 
        """
        # generate n1 samples in normal condition 
        theta_m = np.log2(1e-10 + self.means)
        theta_cov = self._get_covariance_matrix()
        theta = np.random.multivariate_normal(mean=theta_m, cov=theta_cov)
        mu = 2**theta
        if self.data_model == 'poisson':
            args = (mu,)
        else:
            var = self._variance(mu)
            args = ((mu*mu) / (var - mu), mu/var)
        data_dist = self.data_dist(*args)
        x1 = [data_dist.rvs() for _ in range(n1)]
        x1 = np.array(x1)

        # generate n2 samples in test condition 
        fold_changes = np.zeros_like(theta_m) 
        if len(down_idx):
            for i in down_idx:
                fold_changes[i] = -np.log2(np.random.uniform(fold_change_min, fold_change_max))
        if len(up_idx):
            for i in up_idx:
                fold_changes[i] = np.log2(np.random.uniform(fold_change_min, fold_change_max))

        theta = theta + fold_changes 
        mu = 2**theta
        if self.data_model == 'poisson':
            args = (mu,)
        else:
            var = self._variance(mu) 
            args = ((mu*mu) / (var - mu), mu/var)
        data_dist = self.data_dist(*args)
        x2 = [data_dist.rvs() for _ in range(n2)]
        x2 = np.array(x2)

        # stack, shuffle and return 
        x = np.vstack([x1, x2])
        #perm = np.random.permutation(len(x))
        #x = x[perm] 
        return x 
if __name__ == "__main__":
    from pdb import set_trace 
    n = 50
    data = [np.random.poisson(lam=np.random.uniform(50., 100., size=(n,))) for _ in range(200)]
    data = np.array(data)
    g = nx.random_geometric_graph(n, 0.3)
    A = nx.to_numpy_array(g)
    seqgendiff = SeqGenDiff(data=data, data_model='poisson', adj_mat=A)
    X  = seqgendiff.generate_by_network_perturbation(n1 = 200, n2 = 200, )
    print(X.shape)
    set_trace()
