# BrSTNet

## Repository Description:
Breast cancer histopathology image-based gene expression prediction using spatial transcriptomics data and deep learning.

## Abstract:
Tumour heterogeneity in breast cancer poses challenges in predicting outcome and response to therapy. Spatial transcriptomics technologies may address these challenges, as they provide a wealth of information about gene expression at the cell level, but they are expensive, hindering their use in large-scale clinical oncology studies. Predicting gene expression from hematoxylin and eosin stained histology images provides a more affordable alternative for such studies. In this repository, we present the code implementation of BrST-Net, a deep learning framework for predicting gene expression from histopathology images using spatial transcriptomics data. The methodology outperforms previous studies, achieving positive correlations for a larger number of genes and higher correlation coefficients.

## Paper Details:

# Title: "Breast cancer histopathology image-based gene expression prediction using spatial transcriptomics data and deep learning"

Contents:
This repository contains the following Python files:

01_file_organizer.py: Script for organizing the input data files.

02_stain_normalization.py: Script for stain normalization of histology images.

03_spatial_gene_analysis.py: Script for spatial gene analysis using transcriptomics data.

04_generating_train_test.py: Script for generating train and test datasets.

get_patches_with_different_resolution.py: Script for obtaining patches with different resolutions.

enselbl_identity.py: Script for working with Ensembl IDs.

Br_STNet_baseline.py: Main script for the BrST-Net deep learning framework for gene expression prediction.

Please refer to the paper for a detailed explanation of the methodology and the results achieved.
