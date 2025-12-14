[![Project Status: Active …](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lorenzo-costa/pyesbm/HEAD)

# pyesbm

Python code for modeling and inference using [Extended Stochastic Block Models](https://doi.org/10.1214/21-AOAS1595). See the [examples](pyesbm/examples/) for user-friendly illustration.
 
### Key features 

* Estimation and inference for uni-partite and bi-partite graph using Bayesian Non-parametric Stochastic Block-Models

* Complete fitting procedure for collapsed Gibbs sampler using Poisson-Gamma or Beta-Bernoulli models (for the edges) and several instance of Gibbs-type priors (for the clustering structure).

* The code allows the use of categorical and count-valued covariates to "supervise" the Gibbs sampler and borrow information from sources external from the graph.

* Flexible framework that allows users to specify their own model for edges, prior on the clustering partition or covariate types.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/lorenzo-costa/pyesbm.git
cd pyesbm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

To run the example notebooks, install the optional notebook dependencies:
```bash
pip install -e .[notebooks]
```

## Running the example notebooks

After installation:

```bash
jupyter lab
```
Then open any notebook in the `examples/ directory`.

## Repository Structure

```text
├── pyesbm/            # package
│   ├── utilities/        
├── examples/           # notebook examples
├── tests/              # Unit and integration tests
└── README.md           # this file :)
```
---
