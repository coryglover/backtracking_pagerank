# Effects of Backtracking on PageRank

This repository contains the code used to generate and analyze data related to backtracking variants of PageRank. Any questions about this code should be directed to the corresponding author(s) of Glover, Cory, et al. "Effects of Backtracking on PageRank." arXiv preprint arXiv:2211.13353 (2022).

The repository contains three Python Scripts.
The first contains functions helpful for analyzing non-backtracking matrices (nb_general.py).
The second establishes functions used to calculate PageRank, Non-Backtracking PageRank, and Infinite-PageRank as wells as the implementation of Infinite-PageRank clustering (new_pagerank.py).
The third generates PageRank distributions for one graph for various values of $\mu$ (alpha_tools.py).
It also generates plots as seen in Figure 1.

There are additionally two jupyter notebooks.
The first performs analysis on comparing distributions of PageRank values and the second applies infinite-PageRank clustering to generate stochastic block model networks.


## Comparing Distributions
This notebook generates a GNP and HSCM graph, each with 10^4 nodes.
The distributions of standard PageRank and infinite-PageRank and generated and compared visually.

## Infinite-PageRank Clustering
This notebook implements infinite-PageRank clustering on American college football from the year 2000 obtained from http://www-personal.umich.edu/~mejn/netdata/. 
It calculates the associated communities and compares it with the true community labels.
It also generates stochastic block models with three clusters. Each cluster connects within the cluster with probability 0.9 and to each other cluster with probability 0.05. 
The cluster sizes of each community are 28, 30 and 32 respectively.
The accuracy of all applications of the clustering algorithm is computed using NMI.
