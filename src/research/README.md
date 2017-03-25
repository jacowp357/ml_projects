Research Projects
=================

This section contains some research and exploratory code.

Gaussian Mixture Model
======================

One way of modelling continuous vector observations is by using a Gaussian mixture model (GMM). Gaussians are useful continuous mixture components where high density data regions can be observed having more probability mass. The optimal parameters of a GMM can be trained by using maximum likelihood (ML) or alternatively the expectation maximization (EM) algorithm. Popular initialization techniques (to break symmetry) are: (1) set the covariances to be diagonal with large variances, or (2) to use the k-means algorithm and set the means to be the centroids found by it. The latter is used in the `sklearn` package. 

In this research, a GMM model will be used to discretize continuous variables, transforming it to a discrete distribution over the number of Gaussian components. This transformation (from continuous to discrete) is important for random variable compatibility inside a probabilistic graphical model (PGM) framework.  

Latent Dirichlet Allocation
===========================

In standard mixture type models each data-point is assumed to have been generated from a single "cluster". In mixed membership type models a data-point may be a member of more than one "cluster". LDA models consider each data-point to be "explained" by more than one "cluster". The aim is to find common themes among the data assuming that any data-point could potentially contain multiple "overlapping" themes.

Probabilistic Graphical Model
=============================

What makes PGMs an attractive modelling tool is its ability to explicitly take into account uncertainty in a probabilistic framework. In many cases, it is useful to be able to infer some target random variable of interest given incomplete or noisy observations. It is particularly useful to be able to understand the underlying data (and its "construction") by studying the influence of observed random variables on other random variables within the constructs of a graph. Structure learning can be used to exploit the statistical relationships (dependencies) between random variables in a network.  

