# Data Augmentation for Imbalanced Tabular Classification

## Project Overview

This repository contains the experimental implementation for the EECS 6412 research project, *Baseline Evaluation of Data Augmentation for Imbalanced Tabular Classification: Toward Quality-aware Hybrid Augmentation (AQHDA)*, at York University.

The project investigates whether synthetic tabular data improve minority-class prediction while preserving heterogeneous feature types, statistical dependencies, and relationships between features and target labels.

## Methods

Four approaches are evaluated:

- **No Augmentation:** Baseline XGBoost classifier trained on original data.
- **SMOTENC:** Minority-class oversampling for mixed numerical and categorical features.
- **CTGAN:** GAN-based synthetic tabular data generation.
- **TVAE:** Variational autoencoder-based tabular data generation.

The research also explores **AQHDA (Adaptive Quality-aware Hybrid Data Augmentation)**, a preliminary validation-guided framework for selecting synthetic observations based on quality and predictive utility.

## Datasets

Three publicly available datasets are used:

| Dataset | Source |
|---|---|
| Bank Marketing | https://archive.ics.uci.edu/dataset/222 |
| Online Shoppers Purchasing Intention | https://archive.ics.uci.edu/dataset/468 |
| Telco Customer Churn | https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv |

## Experimental Setup

- Stratified data splitting: 60% training, 20% validation, 20% testing.
- Five random seeds: 42, 123, 2024, 2025, 2026.
- Natural class-imbalance ratios are retained.
- CTGAN and TVAE are trained for 30 epochs.
- SMOTENC uses five nearest neighbors.
- XGBoost is used for downstream classification.
- Preprocessing and synthetic generation use training data only.
- Classification thresholds are selected on validation data, with final performance measured on held-out test data.

## Evaluation Metrics

Synthetic data quality is examined through:

1. **Feature-type validity:** Valid categorical values, numerical ranges and integer-like features.
2. **Feature dependencies:** Preservation of numerical and categorical pairwise associations.
3. **Feature–target relationships:** Class-conditional agreement and reference-classifier consistency.
4. **Synthetic diversity:** Exact duplicate rates and novelty of generated samples.
5. **Downstream utility:** Predictive performance after augmentation.

The downstream classification metrics include:

- Average Precision (AP)
- AUROC and ROC curves
- Precision, Recall and F1-score for majority and minority classes separately

Results are summarized across five seeds using means and standard deviations.

## Main Findings

| Dataset | Best mean AP method | AP |
|---|---|---|
| Bank Marketing | No Augmentation | 0.449 |
| Online Shoppers | No Augmentation | 0.744 |
| Telco Churn | TVAE | 0.660 |

For Telco Churn, the no-augmentation baseline achieved AP = 0.657. The TVAE difference was small, and the overlapping variability across

