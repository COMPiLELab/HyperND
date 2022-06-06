## HyperND


This repo contains the implementation for the algorithm HyperND from the paper: 

[Nonlinear Feature Diffusion on Hypergraphs](https://arxiv.org/abs/2103.14867) \
By Konstantin Prokopchik, [Austin R. Benson](https://www.cs.cornell.edu/~arb/) and [Francesco Tudisco](https://ftudisco.gitlab.io/post/) \

To be presented at ICML 2022.


## Baselines


To install required julia packages run `julia packages.jl` or `include(packages.jl)` (if you are in a julia terminal).

We have gathered 5 baselines, each of their realizations is taken from a github page.

1. [APPNP](https://github.com/benedekrozemberczki/APPNP)
2. [HGNN](https://github.com/iMoonLab/HGNN)
3. [HyperGCN](https://github.com/malllabiisc/HyperGCN)
4. [SCE](https://github.com/szzhang17/Sparsest-Cut-Network-Embedding)
5. [SGC](https://github.com/Tiiiger/SGC)

We have wrapped them into packages for convenience, the guide to installation inside `competitors/competitors_setups` directory.

There are 3 experiments:

1. HyperGCN experiment is in the old format in the root folder, that extensively compares HyperGCN with our algorithm.  
  Execute `julia cross_val_datasets_HOLS_ft.jl` or `include("cross_val_datasets_HOLS_ft.jl")` to reproduce.
2. Time experiment compares times of our algorithm and baselines.  
  Execute `julia competitors/scripts/time.jl` or `include("competitors/scripts/time.jl")` to reproduce.
3. Main experiment does a CV for our algorithm and compares the results with all the baselines on the same input data across multiple runs.
  Execute `julia competitors/scripts/main.jl` or `include("competitors/scripts/main.jl")` to reproduce.

The datasets for experiments are inside the `data` folder. Results are stored in `competitors/results`. All the additional information is inside the scripts.
