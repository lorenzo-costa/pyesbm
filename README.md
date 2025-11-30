## pyesbm 

## Problem to solve:
There is not easy to use implementation of Bayesian Non-parametric ESBM. The idea would be to have a package that allows the use to implement their version of ESBM along with some pre-built tools.

At a high level a sbm has:
- likelihood
    Possible choices include
    - Bernoulli 
    - Poisson 
    - Gaussian
- prior 
    Possible choices include:
    - multinomial-dirichlet
    - non-parametric priors
- sampling mechanism:
    -> implement only gibbs sampler but allow the possibility of having different sampling schemes

At a high level I would like to have a class ESBM that allows the user to specify
- likelihood
- prior
- degree-correction or not
- bipartite or not

The class should implement:
- point prediction
- credible balls
- likelihood computation
- fit method
- sampling

