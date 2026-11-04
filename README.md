# Machine Learning Lab 4 — Data Quality Pipeline

## Project Overview

This project implements a reproducible data quality assessment and preparation pipeline using the UCI Adult Income Dataset.

## Objective

The objective is to validate, clean, and transform data before training and evaluating a binary classification model.

## Dataset

* **Source:** OpenML
* **Dataset:** Adult (version 2)
* **Records:** 48,842
* **Columns:** 15
* **Input features:** 14
* **Target:** `class`

## Project Structure

```text
ML_lab4/
├── main.py
├── config.yaml
├── data_card.md
├── README.md
├── requirements.txt
└── results/
    ├── dataset_hash.txt
    ├── validation_report.txt
    ├── model_metrics.json
    ├── label_quality_summary.json
    ├── label_review_sample.csv
    ├── manual_label_review.csv
    ├── preprocessor.joblib
    ├── engineered_preprocessor.joblib
    └── processed_dataset.csv
```

## Pipeline Steps

1. Dataset ingestion from OpenML.
2. Data validation and missing-value inspection.
3. Stratified train/test split (70/30).
4. Duplicate removal from training data.
5. Missing-value imputation.
6. Outlier clipping and numerical scaling.
7. Categorical feature encoding.
8. Feature engineering.
9. Label quality inspection.
10. Logistic Regression model evaluation.
11. SHA-256 hashing and reproducibility documentation.

## Feature Engineering

Two additional features are created:

* `capital_net`: Capital gain minus capital loss.
* `hours_per_age`: Hours per week divided by age minus 16.

## Model Evaluation

Logistic Regression was evaluated using accuracy and F1-score.

| Metric   | Baseline | Engineered |
| -------- | -------: | ---------: |
| Accuracy |   0.8540 |     0.8578 |
| F1-score |   0.6601 |     0.6696 |

Feature engineering improved both metrics.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python main.py
```

## Reproducibility

The project uses:

* `config.yaml` for configuration.
* A fixed random seed.
* Saved preprocessing transformers.
* SHA-256 dataset hashing.
* `data_card.md` for dataset documentation.

All generated reports and results are stored in the `results/` directory.
