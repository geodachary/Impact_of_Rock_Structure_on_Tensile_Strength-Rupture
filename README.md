# Anisotropic Tensile Fracture in Metamorphic Rocks  

### Code and data for analytical and numerical modelling of anisotropic tensile fracture  
#### Acharya & Elsworth  

> **Note:** This repository provides research code and data to support the reproducibility of results presented in the associated publication.  
> The scripts reflect a research workflow and are not distributed as a general-purpose software package.

---

## 1. Overview

This repository contains the numerical models, post-processing scripts, and data used in the study:

> **Tensile and mixed-mode fracture in anisotropic metamorphic rocks:  
> Experiments and anisotropy-aware numerical modelling**

The study integrates laboratory experiments and numerical modelling to investigate tensile and mixed-mode fracture in foliated metamorphic rocks. Specifically, it includes:

- Angle-resolved **Brazilian disc experiments** on augen gneiss and psammitic schist  
- An **anisotropy-aware numerical framework** incorporating:
  - Orthotropic elasticity  
  - Directional tensile strength (tensile cap)  
  - Mesoscale heterogeneity  
  - Foliation spacing effects  
- **Failure-mode classification** (tensile, shear, mixed-mode) and  
  **strain-energy partitioning** under idealised Andersonian stress regimes  

The modelling framework is used to:
1. Reproduce tensile strength as a function of loading angle  
2. Compute stress, displacement, and strain-energy fields  
3. Identify fracture initiation and propagation paths  
4. Generate the figures and tables presented in the manuscript  

---

## 2. Repository structure and workflow

The primary numerical and analytical workflow is implemented in the Jupyter notebook:
 
[Tensile_augen_gneiss.ipynb](https://github.com/geodachary/Impact_of_Rock_Structure_on_Tensile_Strength-Rupture/blob/main/Tensile_augen_gneiss.ipynb).



This notebook contains the complete end-to-end pipeline, including model setup, numerical simulation, failure evaluation, and figure generation for a representative lithology (augen gneiss).

For conciseness, only one rock type is included in this public repository.  
The same modelling approach and workflow were applied to the second lithology (psammitic schist) using identical numerical methods, differing only in material parameters. All results are therefore fully reproducible using the provided scripts.

---

## 3. Data and outputs

Input data are provided in standard text formats, and generated figures are written to the output directories during notebook execution. The file structure mirrors the workflow used in the study to facilitate transparent reproducibility.
