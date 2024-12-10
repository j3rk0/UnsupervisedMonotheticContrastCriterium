# UnsupervisedMonotheticContrastCriterium

python implementation of unsupervised monothetic contrast criterium using minimization of davies-bouldin index as contrast criterium.
DB-index have a different implementation wrt sklearn one to efficiently compute the index for each possible value, this allow to keep umcc computation to O(n^2).

umcc_discretize(data, max_contrast=.45, min_samples=3, scale=True)
is the function to call
numpy only required
