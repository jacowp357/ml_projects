Research projects
=================

This section contains some research and exploratory code.

GMMs
====

One way of modelling continuous vector observations is by using a Gaussian mixture model (GMM). Gaussians are useful continuous mixture components where high density data regions can be observed having more probability mass. The optimal parameters of a GMM can be trained by using maximum likelihood (ML) or alternatively the expectation maximization (EM) algorithm. Popular initialization techniques (to break symmetry) are: (1) set the covariances to be diagonal with large variances, or (2) to use the k-means algorithm and set the means to be the centroids found by it. The latter is used in the `sklearn` package. 

In this research, a GMM model will be used to discretize continuous variables, transforming it to a discrete distribution over the number of Gaussian components. This transformation (from continuous to discrete) is important for random variable compatibility inside a probabilistic graphical model (PGM) framework.  


