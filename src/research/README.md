Research Projects
=================

This section contains some research and exploratory code.

Gaussian Mixture Model
======================

One way of modelling continuous vector observations is by using a Gaussian mixture model (GMM). Gaussian distributions are useful continuous mixture components where high density data regions can be observed having more probability mass. The optimal parameters of a GMM can be trained by using maximum likelihood (ML) or alternatively the expectation maximization (EM) algorithm. Popular initialization techniques (to help break symmetry) are: (1) set the covariances to be diagonal with large variances, or (2) to use the k-means algorithm and set the means to be the centroids found by it. The latter is used in the `sklearn` package. 

In this research, a GMM model will be used to discretize continuous variables, transforming it to a discrete distribution over the number of Gaussian components. This transformation (from continuous to discrete) is important for random variable compatibility inside a discrete probabilistic graphical model (PGM) framework.  

Latent Dirichlet Allocation
===========================

In standard mixture type models each data-point is assumed to have been generated from a single "cluster". In mixed membership type models a data-point may be a member of more than one "cluster". LDA is a generative model that considers sets of data-points to be explained by multiple unobserved "clusters". The unobserved "clusters" contain parts of the data that are similar.

The aim is to find common themes among the data assuming that any data-point could potentially contain multiple "overlapping" themes. These themes can be considered as hidden unobserved random variables forming part of a PGM framework. Hidden variables can play an important role in a PGM network providing additional dynamics to the network and information richness in the form of abstract concepts, rules or constraints (i.e., factors we typically struggle to construct with domain knowledge in a PGM framework).

Probabilistic Graphical Model
=============================

What makes PGMs an attractive modelling tool is its ability to explicitly take into account uncertainty, which is expressed in a probabilistic framework. In many cases, it is useful to be able to infer some target random variable of interest given incomplete or noisy observations. It is particularly useful to be able to understand the underlying data (e.g., in a Bayes network) by studying the influence of observed random variables on other random variables within the constructs of a graph. Structure learning can be used to exploit the statistical relationships (dependencies or independencies) between random variables in a network to help construct a PGM model.

Mutual Information
==================

Mutual information (MI), from the field of Information theory, is a general quantitative measure of the dependence between random variables, or groups of random variables. It is useful in determining whether relationships exist between variables, and is far more capable than standard statistical approaches such as Pearson/Spearman correlation. More specifically, "high" MI between two random variables X and Y can be interpreted as: knowing X tells me something about Y and vice versa. MI is therefore the reduction in uncertainty in Y given that we observe X. Uncertainty in this context is determined by using the calculation of Shannon Entropy H. MI is calculated as follows: I(X;Y) = H(X) + H(Y) - H(X,Y) and is nonnegative (i.e. I(X;Y) ≥ 0) and symmetric (i.e. I(X;Y) = I(Y;X)).