import numpy as np
from bisect import insort


def db_contrast(x):
    """
    compute contrast using davies bouldin index
    
    :param x: SORTED numpy array for which you want to compute resolution
    :return: array of resolution value for each of the element of the input array
    """
    cluster_left_sum = 0  # initialize left partition sum
    cluster_right_sum = x.sum()  # initialize right partition sum
    res = np.ones(x.shape[0])  # initialize scores array
    
    for i in range(1, x.shape[0]-1):
        cluster_left_sum += x[i-1]  # update left partition sum
        cluster_right_sum -= x[i-1] # update right partition sum
        centroid_left = cluster_left_sum / i # compute left centroid  
        centroid_right = cluster_right_sum / ( x.shape[0] - i ) # copmpute right centroid
        intra_dist_left = np.abs( x[:i] - centroid_left).mean()  # compute left mean intra-partition distance
        intra_dist_right = np.abs( x[i:] - centroid_right).mean()  # compute right mean intra-partition distance
        centroid_distance = np.abs(centroid_left - centroid_right)  # compute distance between centroids
        db_score = (intra_dist_left + intra_dist_right) / centroid_distance # compute davies-bouldin index using i as partition edge
        res[i] = db_score  # add result to result array 
    return res



def recursive_partitioning(data,max_contrast,min_samples):
    """
    recursive partitioning algorithm, use an activation stack that contains input data views in order
    to emulate recursive behavor without the risk of exceding python's recursion stack limit.
    
    :param data: sorted array of input data
    :param max_contrast: maximum contrast value to perform a split
    :param min_samples: minimum number of samples in a partition
    :return: array of partition edges
    """
    res = []
    stack = [data]  # activation stack containing partition indexes
    while stack:   # while stack is not empty
        curr=stack.pop(-1)  # pop stack
        if curr.shape[0] > 2 * min_samples:        
            scores = db_contrast(curr)  # compute contrast for each treshold
            idx = np.argmin(scores[min_samples+1:-min_samples+1]) + min_samples + 1  # get index of the best treshold
            t, contrast = curr[idx], scores[idx]
            
            if contrast < max_contrast or len(res)<2:  # check stop conditions but return at least 2 partition
                insort(res, t)  # add treshold to sorted edges array
                stack.append(curr[curr<t])  # add new recursion levels to stack
                stack.append(curr[curr>t])
    return res
                
            


def umcc_discretize(data, max_contrast=.45, min_samples=3, scale=True):
    """
    Unsupervised Monothetic Contrast Criterium (UMCC) discretization of continuous data 
    using a contrast-based recursive partitioning approach.
    This implementation uses davies-bouldin index as contrast metric.
    
    :param data: input array to discretize.
    :param max_contrast: maximum contrast value to consider a split.
    :param min_samples: minimum number of samples to consider for a split.
    :param scale: True if you want to scale the discretized data by the number of bins.
    :return: discretized input array.
    """
    sorted_data = np.sort(data) # sort input array
    edges =  recursive_partitioning(sorted_data, max_contrast, min_samples) # get bin edges using umcc
    ret = np.digitize(data, edges) # digitize result according to the umcc bins
    return ret/ret.max() if scale else ret

