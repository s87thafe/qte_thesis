# QTE Thesis

This repository contains the codebase, utilities, and simulations for the Master’s Thesis 
"Inference for Quantile Treatment Effects in High-Dimensional Settings", in partial fulfillment of the requirements for the degree os Master of Science at the University of Bonn.

The project is structured into two main components:

- **analysis/**  
  Contains core package implementations, including the orthogonal score estimator 
  and the weighted double selection estimator (Belloni, Chernozhukov, and Kato, 2019).  
  These modules are written for general use and can be adapted for related research.

- **final/**  
  Includes project-specific simulations, empirical estimations (conducted on the Marvin-Cluster of the University of Bonn), 
  as well as plotting scripts and table creation utilities for thesis results.

---

## Getting Started

1. Create the environment `conda env create -f environment.yml`
2. Activate environment via `conda activate qte_thesis`
3. Install dependencies `pip install -e .`
