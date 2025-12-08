# pyesbm

Python code for modeling and inference using [Extended Stochastic Block Models](https://doi.org/10.1214/21-AOAS1595). See the [examples](pyesbm/examples/) for user-friendly illustration.
 
### Key features 

* Estimation and inference for uni-partite and bi-partite graph using Bayesian Non-parametric Stochastic Block-Models

* Complete fitting procedure for collapsed Gibbs sampler using Poisson-Gamma or Beta-Bernoulli models (for the edges) and several instance of Gibbs-type priors (for the clustering structure).

* The code allows the use of categorical and count-valued covariates to "supervise" the Gibbs sampler and borrow information from sources external from the graph.

* Flexible framework that allows users to specify their own model for edges, prior on the clustering partition or covariate types.




