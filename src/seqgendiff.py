import numpy as np
from scipy import stats

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
        self.cov = cov 
        self.data_model = data_model.strip().lower()
        self.data_dist = stats.poisson if self.data_model == 'poisson' else stats.nbinom
    
    def generate_by_network_perturbation(self, n1, n2, 
                                down_idx=[], up_idx=[], 
                                fold_change_min=2, fold_change_max=5):
    
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
        if self.cov == None:
            theta_cov = np.eye(len(theta_m))
        else:
            theta_cov = self.cov
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
        perm = np.random.permutation(len(x))
        x = x[perm] 
        return x 