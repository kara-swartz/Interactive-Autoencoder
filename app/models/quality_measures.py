"""
Quality metrics for dimensionality reduction.
"""

import math
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def qm_mae_dist(original, reduced):
    """
    Mean absolute error of distances
    """
    hd_dists = euclidean_distances(original)
    ld_dists = euclidean_distances(reduced)
    total_absolute_error = np.sum(np.abs(hd_dists - ld_dists))
    return total_absolute_error / original.shape[0]
    
    
    
def qm_mse_dist(original, reduced):
    """
    Mean squared error of distances
    """
    hd_dists = euclidean_distances(original)
    ld_dists = euclidean_distances(reduced)
    total_squared_error = np.sum((hd_dists - ld_dists) ** 2)
    return total_squared_error / original.shape[0]
    
    

def qm_corr_dist(original, reduced):
    """
    Correlation of distances
    """
    hd_dists = euclidean_distances(original)
    ld_dists = euclidean_distances(reduced)
    return np.corrcoef(hd_dists.flatten(), ld_dists.flatten())[0,1]

