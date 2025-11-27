
# Anisotropic Tensile Fracture in Metamorphic Rocks  
Code & Data for Acharya & Elsworth (submitted to *Theoretical and Applied Fracture Mechanics*)

> **Note:** This is research code provided to support reproducibility of the published results.  
> It is not a polished software package and may require minor adjustments for different systems.

------------------------------------------------------------------------------------------------------


## 1. Overview

This repository contains the scripts and data used in the paper:

> **Tensile and mixed-mode fracture in anisotropic metamorphic rocks:  
> Experiments and anisotropy-aware numerical modelling**

The work combines:

- **Angle-resolved Brazilian disc experiments** on two foliated lithologies  
  – Augen gneiss  
  – Psammitic schist  
- An **anisotropy-aware numerical framework** that includes:
  - Orthotropic elasticity
  - Directional tensile strength (“tensile cap”)
  - Mesoscale heterogeneity
  - Foliation spacing effects
- **Failure analysis** (tensile, shear, mixed-mode) and  
  **strain-energy partitioning** under idealised Andersonian stress orientations.

The code here is designed to:

1. Reproduce the **tensile strength vs loading angle** curves.  
2. Compute **stress, displacement, and energy fields** in Brazilian discs.  
3. Classify **failure modes** and extract **crack paths**.  
4. Generate the **figures and tables** used in the manuscript.