# scPOEM

## Model Overview
scPOEM provides a workflow to jointly embed chromatin accessibility peaks and expressed genes into a shared low-dimensional space using paired single-cell ATAC-seq (scATAC-seq) and single-cell RNA-seq (scRNA-seq) data. It integrates regulatory relationships among peak-peak interactions (via Cicero), peak-gene interactions (via Lasso, random forest, and XGBoost), and gene-gene interactions (via principal component regression). With the input of paired scATAC-seq and scRNA-seq data matrices, scPOEM assigns a low-dimensional feature vector to each gene and peak. Additionally, it supports the reconstruction of gene-gene network with low-dimensional projections (via $\epsilon$-NN) and then the comparison of the networks of two conditions through manifold alignment implemented in scTenifoldNet.

### Notes:  
- The current implementation of scPOEM utilizes [Monocle3](https://cole-trapnell-lab.github.io/monocle3/docs/installation/) and [Cicero for Monocle3](https://cole-trapnell-lab.github.io/cicero-release/docs_m3/) to construct peak-peak networks. We are in the process of releasing the **Monocle2**-based version, as an R package on CRAN.
---

## Install
scPOEM/R is currently a development version, but can be installed and used with the following command:
```R
library(remotes)
install_github('Houyt23/scPOEM')
library(scPOEM)
```
---

## Recommended environment
This package is implemented using both R and Python. To ensure proper functionality, we recommend the following environment settings:
- **R version**: ≥ 4.1.0  
- **Python version**: 3.9 
- **Python environment**: It is recommended to use a Conda environment for Python dependencies management.  
  You can create a conda environment using:
  ```bash
  conda create -n scPOEM_env python=3.9
  conda activate scPOEM_env
  ```
  Then install the required Python packages:
  ```bash
  pip install os random numpy scipy scikit-learn matplotlib tqdm ray tensorflow
  ```
---

## Input Data Requirements

| Example File          | Description                                |
|---------------------|--------------------------------------------|
| `X.mtx`| Preprocessed chromatin accessibility data|
| `Y.mtx`| Preprocessed Gene expression data|
| `peak_data.csv`| Genomic coordinates of ATAC peaks|
| `gene_data.csv`| Gene names|
| `cell_data.csv`| Cell identifiers|
| `peakuse_100kbp.csv` | Peak IDs within a certain range (e.g. 100kbp) of each gene|
| `[species].genome`   | Chromosome sizes|

---

## Available functions
| Function | Description |
|--------|--------|
| `GGN` | Construct the gene-gene network. |
| `PPN` | Construct peak-peak network. |
| `PGN_Lasso` | Construct the peak-gene network via Lasso. |
| `PGN_RF` | Construct the peak-gene network via random forest. |
| `PGN_XGBoost` | Construct the peak-gene network via XGBoost. |
| `pg_embedding` | Learn the low-dimensional representations for peaks and genes with a meta-path based method. |
| `align_embedding` | Reconstruct gene networks via $\epsilon$-NN and compare conditions using manifold alignment implemented in scTenifoldNet. *|
| `scPOEM` | Starting from paired scATAC-seq and scRNA-seq data, it first construct regulatory networks—including peak-peak, peak-gene and gene-gene interactions, and then embed peaks and genes into a shared low-dimensional space. It supports both **single** and **compare** modes, allowing analysis within a single condition or differential comparison across conditions. |

### Notes:
- '*' means only utilized in **compare** mode.

---

## Ouput

### Single mode

Returns a list containing the following elements:
- `E`: Low-dimensional representations of peaks and genes.  
- `peak_node`: Peak IDs that are associated with other peaks or genes (used in pg_embedding).  
- `gene_node`: Gene IDs that are associated with other peaks or genes (used in pg_embedding).

### Compare mode

Returns a list consisting of three sublists:
- The single-mode result for the first condition.
- The single-mode result for the second condition.
- A summary list, which includes:
   - `E_g2`: Low-dimensional embedding representations of genes under the two conditions after align_embedding.  
   - `common_genes`: Genes shared between both conditions.  
   - `diffRegulation`: A list of differential regulatory information for each gene.

---

## Quick-Start Examples
Two minimal examples to demonstrate the workflow:

⚠️ **Note**: These are **toy examples** for architecture verification. For real biological data used in our paper, please see `/real_data`.

### Example 1: Single Dataset Analysis
```R
library(scPOEM)
dirpath <- "./example_data"
# An example for analysing a single dataset.
# Download and read data.
data(example_data_single)
single_result <- scPOEM(mode = "single",
                        input_data=example_data_single,
                        dirpath=file.path(dirpath, "single"))
```

### Example 2: Two States Comparative Analysis
```R
library(scPOEM)
dirpath <- "./example_data"
# An example for analysing and comparing datasets from two conditions.
# Download compare mode example data
data(example_data_compare)
compare_result <- scPOEM(mode = "compare",
                         input_data=example_data_compare,
                         dirpath=file.path(dirpath, "compare"))
```

---
⚠️  **For more details, please check the help document available in the inst/doc/ folder of the repository.**
