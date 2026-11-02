# Data Augmentation for Imbalanced Tabular Classification

**Baseline Evaluation of Data Augmentation for Imbalanced Tabular Classification: Toward Quality-aware Hybrid Augmentation (AQHDA)**

**Institution:** York University  
**Course:** EECS 6412 — Data Mining  
**Project Type:** Research Project

## 1. Project Overview

This repository contains the implementation and experimental evaluation of tabular data augmentation methods for imbalanced binary classification.

The main objective is to investigate whether synthetic training examples improve minority-class prediction while preserving the statistical and structural properties of the original tabular data.

Particular attention is given to:

- Preserving heterogeneous numerical and categorical feature types.
- Maintaining dependencies and relationships among features.
- Preserving relationships between features and target labels, particularly for minority-class observations.
- Evaluating synthetic data quality and diversity.
- Measuring downstream classification performance under class imbalance.

The project also investigates **AQHDA (Adaptive Quality-aware Hybrid Data Augmentation)**, a proposed framework for selecting and refining synthetic observations based on quality criteria and validation feedback.

## 2. Research Objectives

The research addresses the following questions:

1. Do existing tabular data augmentation methods improve classification performance compared with training on original data alone?
2. How effectively do different augmentation methods preserve mixed feature types and relationships among variables?
3. Do synthetic minority-class samples maintain realistic feature–target relationships?
4. How do statistical quality and synthetic diversity relate to downstream predictive utility?
5. Can quality-aware synthetic sample selection improve upon conventional augmentation approaches?

The broader goal is to develop a reliable augmentation strategy that prioritizes synthetic sample quality rather than simply increasing training-set size.

## 3. Methods

Four experimental approaches are evaluated.

### 3.1 No Augmentation (Baseline)

An XGBoost classifier is trained using the original training data without synthetic observations.

This provides a reference for determining whether augmentation improves downstream performance.

### 3.2 SMOTENC

Synthetic Minority Over-sampling Technique for Nominal and Continuous features (SMOTENC) generates minority-class observations using interpolation and categorical-feature handling.

- Supports numerical and categorical features.
- Uses five nearest neighbors.
- Provides a conventional interpolation-based augmentation baseline.

### 3.3 CTGAN

Conditional Tabular Generative Adversarial Network (CTGAN) is a GAN-based generative model designed for tabular data.

- Models numerical and categorical feature distributions.
- Can capture nonlinear patterns among variables.
- Is trained on minority-class observations for the augmentation experiments.

### 3.4 TVAE

Tabular Variational Autoencoder (TVAE) is a probabilistic generative model that learns latent representations of tabular observations.

- Supports heterogeneous numerical and categorical features.
- Generates synthetic observations through an encoder–decoder framework.
- Is trained on minority-class observations.

### 3.5 Proposed AQHDA Extension

AQHDA investigates a quality-aware approach to synthetic sample selection.

The preliminary framework combines candidate synthetic observations with quality and diversity considerations and uses validation feedback to guide selection.

The implementation explores progressive selection and Optuna-based optimization.

AQHDA is an exploratory research extension rather than a fully validated replacement for the baseline methods.

## 4. Datasets

Three publicly available tabular datasets are used.

| Dataset | Source | Classification target |
|---|---|---|
| Bank Marketing | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222) | Term-deposit subscription |
| Online Shoppers Purchasing Intention | [UCI Online Shoppers](https://archive.ics.uci.edu/dataset/468) | Purchase intention |
| Telco Customer Churn | [Public IBM sample dataset mirror](https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv) | Customer churn |

The experiments retain the natural class distributions rather than imposing a fixed 90/10 imbalance ratio.

The Bank Marketing call-duration feature is removed because it is unavailable before the outcome and may introduce target leakage.

## 5. Experimental Setup

### Data Partitioning

Each dataset is divided using stratified sampling:

| Partition | Proportion |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

Five random seeds are used:

`42, 123, 2024, 2025, 2026`

The same partitions are used to compare augmentation methods fairly.

Preprocessing and synthetic generation use training data only. Validation data are used for threshold selection and validation-guided optimization, while test data are reserved for final evaluation.

### Augmentation Settings

- SMOTENC uses five nearest neighbors.
- CTGAN is trained for 30 epochs.
- TVAE is trained for 30 epochs.
- Each baseline augmentation method generates the same number of synthetic minority-class observations.
- Synthetic sample generation is restricted to the training data.

The number of added minority observations is defined as:

`N = min(n_majority - n_minority, n_minority)`

This limits the augmentation budget and does not necessarily produce fully balanced datasets.

### Downstream Classification

XGBoost is used as the downstream classifier, with:

- Number of trees: 150
- Maximum depth: 4
- Learning rate: 0.05

Classification thresholds are selected using validation data to maximize minority-class F1-score.

## 6. Synthetic Data Quality Evaluation

Synthetic minority-class observations are compared with real minority-class training observations.

### 6.1 Heterogeneous Feature-Type Preservation

Checks whether synthetic samples preserve valid representations of numerical and categorical features.

Evaluations include:

- Invalid categorical values
- Non-finite numerical values
- Out-of-range numerical values
- Violations of integer-like feature constraints

### 6.2 Inter-Feature Dependency Preservation

Evaluates whether statistical relationships among features are maintained.

Measures include:

- Pearson correlation for numerical relationships
- Cramér's V for categorical associations
- Correlation ratio for numerical–categorical relationships

### 6.3 Feature–Target Relationship Preservation

Evaluates whether generated minority-class observations remain consistent with class-conditional characteristics of real data.

Checks include:

- Class-conditional numerical statistics
- Categorical distribution differences
- Reference-classifier probability comparisons

These measures are proxies for feature–target consistency and do not establish that every generated label is correct.

### 6.4 Synthetic Data Diversity

Evaluates whether augmentation produces sufficiently varied observations.

The reported evaluation includes exact-duplicate checks against real and synthetic observations.

Exact-duplicate rates provide a limited measure of diversity and do not capture all forms of mode collapse or synthetic redundancy.

### 6.5 Minority-Class Quality

Examines the validity, statistical agreement and reference-model consistency of synthetic minority observations relative to real minority training data.

This is a targeted interpretation of the quality checks above rather than an additional independently validated metric.

### 6.6 Downstream Predictive Utility

Evaluates whether synthetic augmentation improves classification on held-out test data.

The principal threshold-independent metrics are:

- Average Precision (AP)
- Area Under the ROC Curve (AUROC)

Additional class-specific metrics include:

- Precision
- Recall
- F1-score

Precision, Recall and F1-score are reported separately for the majority and minority classes.

Results are summarized across five seeds using mean and standard deviation.

## 7. Experimental Results

The following results are from the initial midterm experiments.

| Dataset | Best method by mean AP | Mean AP |
|---|---|---:|
| Bank Marketing | No Augmentation | 0.449 |
| Online Shoppers | No Augmentation | 0.744 |
| Telco Customer Churn | TVAE | 0.660 |

### Key Findings

**Bank Marketing:** No augmentation achieved the highest mean AP of 0.449.

**Online Shoppers:** No augmentation achieved the highest mean AP of 0.744.

**Telco Customer Churn:** TVAE achieved a mean AP of 0.660, compared with 0.657 without augmentation.

The Telco difference was small, and variation across the five experimental seeds does not establish a statistically reliable improvement.

Overall, the initial experiments suggest that generating more synthetic observations does not automatically improve downstream classification.

These findings motivate investigating quality-aware selection of synthetic samples.

**Important:** The Telco results are preliminary because of a documented `TotalCharges` preprocessing issue. They require correction and rerunning before being considered fully validated.

## 8. AQHDA Research Extension

The proposed AQHDA framework investigates two complementary components.

### Stage 1: Quality-aware Synthetic Sample Selection

Candidate observations from augmentation methods are evaluated using:

- Feature-type validity
- Inter-feature dependency preservation
- Feature–target consistency
- Statistical quality
- Diversity

The objective is to avoid including synthetic observations solely because they were generated successfully.

### Stage 2: Validation-guided Refinement

Validation feedback is used to investigate sample-selection strategies and augmentation configurations.

The preliminary implementation uses progressive sample selection and Optuna-based optimization.

This stage aims to determine whether selected synthetic observations improve downstream classification without compromising statistical quality.

The preliminary results should not be interpreted as conclusive evidence that AQHDA outperforms the baseline approaches.

## 9. Code and Reproducibility

### GitHub Repository

https://github.com/atizahiri/Data-Augmentation-for-Tabular-Data

### Main Notebook

[Tabular_Augmentation_Baselines_and_AQHDA.ipynb](https://github.com/atizahiri/Data-Augmentation-for-Tabular-Data/blob/main/Tabular_Augmentation_Baselines_and_AQHDA.ipynb)

### Required Libraries

The repository provides `requirements.txt`, listing the main Python dependencies:

- numpy, pandas, scipy
- scikit-learn, imbalanced-learn
- ctgan, xgboost, ucimlrepo
- matplotlib, seaborn
- torch, optuna

### Running the Experiments

1. Download or open the notebook from GitHub.
2. Open the notebook in Google Colab.
3. Install the required dependencies using the repository's `requirements.txt` or the notebook's setup cells.
4. Run the notebook cells sequentially.
5. Examine the generated synthetic-quality summaries, classification tables and ROC plots.

An internet connection is required because the public datasets are retrieved at runtime.

For a fresh environment, dependencies can be installed using:

```bash
pip install -r requirements.txt
```

The notebook's complete execution has not been independently verified in a clean environment. Dependency versions are not pinned, so results may depend on installed package versions.

## 10. Limitations

The current experimental study has several limitations:

- Only three publicly available datasets are included.
- Five repeated splits provide a preliminary assessment but do not establish statistical significance.
- CTGAN and TVAE are trained for only 30 epochs.
- Generative-model training may not be fully deterministic.
- Pairwise dependence measures do not capture all higher-order relationships.
- Feature–target consistency checks are indirect proxies rather than proof of label correctness.
- Exact duplicate rates provide only a partial assessment of diversity.
- Some integer-like synthetic values are not repaired or rounded.
- The Telco `TotalCharges` preprocessing issue requires correction and rerunning.
- AQHDA remains a preliminary research extension requiring further evaluation.

## 11. Research Status and Future Work

This repository contains the baseline implementation and preliminary AQHDA experiments.

Planned future work includes:

- Correcting and rerunning affected experiments.
- Improving synthetic sample validity and dependency preservation.
- Investigating stronger quality-aware selection criteria.
- Refining validation-guided synthetic sample selection.
- Evaluating additional datasets and more challenging limited-data settings.
- Comparing performance under different imbalance ratios and synthetic sample budgets.
- Extending the analysis to additional generative augmentation methods.

The longer-term objective is to develop a reproducible and statistically grounded hybrid augmentation framework that improves minority-class prediction without sacrificing synthetic data quality.

## 12. References

1. Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research.
2. Xu et al. (2019). *Modeling Tabular Data using Conditional GAN*. NeurIPS.
3. Kotelnikov et al. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. ICML.
4. Borgwardt et al. (2023). *Language Models are Realistic Tabular Data Generators*. ICLR.
5. Chen and Guestrin (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
6. Saito and Rehmsmeier (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE.

## 13. Academic Context

This repository supports an academic research project conducted at York University.

The experimental results represent a midterm-stage investigation. Further corrections, validation and methodological development are required before drawing definitive conclusions about the proposed AQHDA framework.
