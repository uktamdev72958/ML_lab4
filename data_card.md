# Adult Income Dataset — Data Card

## 1. Dataset Overview

* **Dataset name:** Adult Income Dataset
* **Source:** OpenML
* **Dataset identifier:** Adult, version 2
* **Total records:** 48,842
* **Total columns:** 15
* **Input features:** 14
* **Target column:** `class`
* **Machine learning task:** Binary classification

## 2. Purpose

The objective of this project is to build a reproducible data preparation pipeline and evaluate a baseline machine learning model.

The model predicts whether a person's annual income exceeds $50,000.

## 3. Target Labels

The target column is `class`.

* `<=50K`: Annual income is at most $50,000.
* `>50K`: Annual income exceeds $50,000.

## 4. Data Quality

The initial dataset contains missing values in three columns:

* `workclass`: 2,799 missing values
* `occupation`: 2,809 missing values
* `native-country`: 857 missing values

The validation pipeline checks dataset dimensions, required columns, numerical ranges, missing values, and target labels.

The validation run found no values outside the configured numerical ranges and no invalid target labels.

## 5. Data Preparation

The pipeline performs the following operations:

1. Splits the dataset into training and testing sets using stratification.
2. Removes exact duplicate records from the training data only.
3. Imputes missing numerical values using the median.
4. Imputes missing categorical values using the most frequent value.
5. Clips numerical outliers using the training data's 1st and 99th percentiles.
6. Scales numerical features.
7. One-hot encodes categorical features.

The preprocessing transformer is fitted on the training data and then applied to the test data.

## 6. Feature Engineering

Two additional features are created:

* `capital_net`: Capital gain minus capital loss.
* `hours_per_age`: Hours worked per week divided by age minus 16.

The project compares Logistic Regression models trained with and without these additional features.

## 7. Evaluation

The models are evaluated using accuracy and F1 score.

Initial results:

| Metric   | Baseline | Engineered Features |
| -------- | -------: | ------------------: |
| Accuracy |   0.8540 |              0.8578 |
| F1 Score |   0.6601 |              0.6696 |

These results indicate a modest improvement after feature engineering.

## 8. Known Limitations and Bias

The dataset contains demographic and employment-related attributes. These variables may reflect historical social and economic inequalities.

The dataset may not represent every population or current labor market conditions. Model predictions should not be interpreted as definitive judgments about individuals.

The dataset also contains missing values and groups of identical input features with conflicting target labels. These issues require further investigation.

## 9. Privacy and Responsible Use

The data contains demographic and employment-related information. It should be used for educational and analytical purposes with appropriate care.

Predictions should not be used as the sole basis for consequential decisions about individuals.

## 10. Reproducibility

The project includes:

* `main.py`: Main data preparation and evaluation pipeline.
* `config.yaml`: Pipeline configuration.
* `results/validation_report.txt`: Validation report.
* `results/preprocessor.joblib`: Saved baseline preprocessor.
* `results/engineered_preprocessor.joblib`: Saved engineered-feature preprocessor.
* `results/processed_dataset.csv`: Processed training dataset.
* `results/dataset_hash.txt`: SHA-256 hash.
* `results/model_metrics.json`: Model evaluation metrics.

The pipeline uses a fixed random seed for the train/test split and model initialization.

## 11. License and Source Notes

The dataset is retrieved through OpenML using the dataset name `adult` and version 2.

Before redistributing the dataset or using it outside this educational project, verify the applicable dataset license and usage terms from the original source.

### 12. Manual Label Quality Review

A random sample of 50 records was manually inspected for missing values, unusual numerical values, inconsistent attributes, and potentially incorrect target labels.

No clearly impossible values or definite target-label errors were identified during the review. Some records contained missing values in the `workclass` and `occupation` columns, which were consistent with the missing values identified during the initial validation stage.

Unusual but valid values, such as a capital gain of 99,999, were not considered errors because they fall within the accepted dataset ranges.

The target labels could not be independently verified because actual annual income values were not available in the reviewed records.

**Conclusion:** No definite labeling errors were identified in the reviewed sample. However, complete label verification would require access to the original income information.
