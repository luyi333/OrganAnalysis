# OrganAnalysis
This repository contains a Python implementation for comprehensive analysis of organs based on CT imaging.

# Project Documentation

## Data Preparation

The preparation of multi-organ segmentation data follows the guidelines provided by nnUNet, which can be accessed at [[nnUNet link](https://github.com/MIC-DKFZ/nnUNet)]. In addition to the original resolution images, a set of data prepared at a lower resolution is required for the low-resolution segmentation branch.

## Segmentation Workflow

### Stage One: Coarse Segmentation

The first stage involves coarse segmentation, which can be executed using the script located at `/AIG-MSPU/run/run_training.py`.

### Stage Two: Fusion of Segmentation Results

After completing the parallel segmentation in the first stage, the fusion of segmentation results is carried out using the script found at `/AIG-MSPU/Fusion/train.py`.

## Material Decomposition

For an example of standard SECT-based material decomposition, refer to `/DecomposeMaterials/decompose_single.py`.
