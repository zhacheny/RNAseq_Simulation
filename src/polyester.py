import numpy as np 

class Polyester:
    """
    An implementation of the Polyester synthetic RNASeq data simulation
    procedure. 

    Under this model, the read count of each gene is sampled independently through a negative 
    binomial distribution, y_ij \sim NegBin(mu_ij, r_ij), where i indexes the gene and 
    j indexes the condition. 

    The mean mu_ij is sampled from real data. The method takes as input, a csv file
    of real RNASeq data, where columns are genes and and rows are samples and the mean 
    of the read counts along each column is assigned to mu. 

    Reference:
    Frazee, A. C., Jaffe, A. E., Langmead, B., & Leek, J. T. (2015). 
    Polyester: simulating RNA-seq datasets with differential transcript 
    expression. Bioinformatics, 31(17), 2778-2784.
    """
    def _variance(self, mu):
        r = mu / self.disp_factor
        return mu + ((mu**2) / r)
    
    def __init__(self, data, disp_factor=3):
        """
        Parameters
        ----------
        data <numpy.ndarray> - numpy matrix of read data - rows are samples columns are genes. 
        disp_factor <float> - The factor by which to scale the mean to obtain 
                              dispersion factor of the negative binomial distribution
                            for sampling. dispersion factor r = mean / disp_factor
                            default: disp_factor = 3. 
                            Variance of the negative 
        """" 
        assert data.ndim == 2
        means = data.mean(0)  # per gene mean
        self.disp_factor = disp_factor
        self.mu = means

    def _generate(self, num_samples, mu, var):
        p = mu / var  
        n = (mu**2) / (var - mu)
        return np.array([np.random.negative_binomial(n, p) \
                        for _ in range(num_samples)])

    def generate(self, n1, n2, down_idx=[], up_idx=[], fold_change_min=2, fold_change_max=5):
        """
        Simulate RNASeq data for two conditions. 
        The baseline control condition has means determined by the input 
        data. 
        The case condition has means that are modified according to randomly generated 
        fold change sampled from lambda \sim unif(fold_change_min, fold_change_max)
        
        Parameters:
        ----------
        n1 <int> - Number of samples in control condition. 
        n2 <int> - Number of sampels in case condition. 
        down_idx <list of int> - A list of indexes for down regulated genes. 
        up_idx <list of int> - A list of indexes for up regulated genes. 
        fold_change_min <float> - Minimum fold change.
        fold_change_max <float> - Max. fold change. 
        """

        #first generate n1 samples in normal condition. 
        mu = self.means 
        var = self._variance(mu)
        x1 = self._generate(n1, mu, var)

        # modify vector of means 
        fold_changes = np.ones_like(self.mu)
        if len(up_idx):
            lam = np.random.uniform(
                        low=fold_change_min, 
                        high=fold_change_max, 
                        size=(len(up_idx))
                                )
            fold_changes[up_idx] = lam
        
        if len(down_idx):
            lam = np.random.uniform(
                        low=fold_change_min, 
                        high=fold_change_max, 
                        size=(len(up_idx))
                                )
            fold_changes[up_idx] = 1./lam

        
        # generate n2 samples of test condition 
        mu_new = mu*fold_changes
        var_new = self._variance(mu_new)
        x2 = self._generate(n2, mu_new, var_new)

        # concat, shuffle and return. 
        X = np.vstack([x1, x2])
        perm = np.random.permutation(len(X))
        X = X[perm] 
        return X