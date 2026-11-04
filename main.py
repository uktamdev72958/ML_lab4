import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


EXPECTED_RANGES = {
    "age": (17, 90),
    "hours-per-week": (1, 99),
    "capital-gain": (0, 99999),
    "capital-loss": (0, 4356),
}
VALID_TARGETS = {"<=50K", ">50K"}


def load_config(config_path="config.yaml"):
    """Load pipeline settings from a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print("Configuration loaded successfully.")
    return config


def load_dataset(config):
    """Load the configured Adult dataset from OpenML."""
    dataset_config = config["dataset"]
    print("Loading the Adult dataset...")
    adult = fetch_openml(
        name=dataset_config["name"],
        version=dataset_config["version"],
        as_frame=True,
    )
    df = adult.frame
    print("Dataset loaded successfully!")
    return df


def validate_dataset_structure(df, config):
    """Check the dataset dimensions, target column, and column names."""
    dataset_config = config["dataset"]
    expected_shape = (
        dataset_config["expected_rows"],
        dataset_config["expected_columns"],
    )
    if df.shape != expected_shape:
        raise ValueError(f"Unexpected dataset shape: {df.shape}; expected {expected_shape}")

    target_column = dataset_config["target_column"]
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names detected.")
    print("Basic structural checks passed.")


def validate_data(df, target_column):
    """Validate numerical ranges, missing values, and target labels."""
    print("\n--- Data Validation Report ---")
    errors = []
    required_columns = set(EXPECTED_RANGES) | {target_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    for column, (minimum, maximum) in EXPECTED_RANGES.items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.notna() & ((values < minimum) | (values > maximum))
        count = int(invalid.sum())
        print(f"{column}: {count} values outside [{minimum}, {maximum}]")
        if count:
            errors.append(f"{column}: {count} out-of-range values")

    missing_counts = df.isna().sum()
    print("\nMissing values by column:")
    print(missing_counts[missing_counts > 0].to_string() if missing_counts.any() else "No missing values")

    if target_column in df.columns:
        target = df[target_column].astype("string").str.strip()
        invalid_targets = ~target.isin(VALID_TARGETS) & target.notna()
        invalid_count = int(invalid_targets.sum())
        print(f"\nInvalid target labels: {invalid_count}")
        if invalid_count:
            errors.append("Unexpected target labels detected")

    question_mark_counts = {}
    text_columns = df.select_dtypes(include=["object", "category", "string"]).columns
    for column in text_columns:
        count = int(df[column].astype("string").str.strip().eq("?").sum())
        if count:
            question_mark_counts[column] = count
    print("\nColumns containing '?' markers:")
    print(question_mark_counts if question_mark_counts else "None")

    print("\nValidation status:")
    if errors:
        print("ISSUES FOUND")
        for error in errors:
            print("-", error)
    else:
        print("No validation errors found in the checks performed.")

    return {"errors": errors, "missing_values": missing_counts.to_dict(), "shape": df.shape}


def save_validation_report(df, report_path):
    """Save a summary of dataset validation checks."""
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("ADULT DATASET VALIDATION REPORT\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\n\n")
        file.write("DATA TYPES\n" + "-" * 40 + "\n")
        file.write(df.dtypes.to_string() + "\n\n")
        file.write("MISSING VALUES\n" + "-" * 40 + "\n")
        file.write(df.isna().sum().to_string() + "\n\n")
        file.write("DUPLICATE ROWS\n" + "-" * 40 + "\n")
        file.write(f"Exact duplicate rows: {df.duplicated().sum()}\n")
    print(f"\nValidation report saved to: {report_path}")


def describe_problem(df, target_column):
    """Display dataset and ML problem information."""
    print("\n--- Dataset Information ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Input features: {df.shape[1] - 1}")
    print(f"Target column: {target_column}")
    print("Problem type: Binary classification")
    print("Goal: Predict whether annual income exceeds $50,000.")


def split_dataset(df, config):
    """Split data using settings from config.yaml."""
    print("\n--- Train-Test Split ---")
    target_column = config["dataset"]["target_column"]
    split_config = config["split"]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=y if split_config["stratify"] else None,
    )
    print(f"Training features: {X_train.shape}")
    print(f"Testing features: {X_test.shape}")
    print("\nTraining target distribution:")
    print(y_train.value_counts(normalize=True).round(3))
    print("\nTesting target distribution:")
    print(y_test.value_counts(normalize=True).round(3))
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test, config):
    """Impute, clip, scale, and encode using training data only."""
    print("\n--- Data Preprocessing ---")
    preprocessing_config = config["preprocessing"]

    numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()
    print("Numerical columns:", numerical_columns)
    print("Categorical columns:", categorical_columns)

    lower_q = preprocessing_config["outlier_lower_quantile"]
    upper_q = preprocessing_config["outlier_upper_quantile"]
    lower_limits = X_train[numerical_columns].quantile(lower_q)
    upper_limits = X_train[numerical_columns].quantile(upper_q)

    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    X_train_clean[numerical_columns] = X_train_clean[numerical_columns].clip(
        lower=lower_limits, upper=upper_limits, axis=1
    )
    X_test_clean[numerical_columns] = X_test_clean[numerical_columns].clip(
        lower=lower_limits, upper=upper_limits, axis=1
    )

    numerical_steps = [
        ("imputer", SimpleImputer(strategy=preprocessing_config["numerical_imputation"]))
    ]
    if preprocessing_config.get("scale_numerical_features", True):
        numerical_steps.append(("scaler", StandardScaler()))
    numerical_pipeline = Pipeline(numerical_steps)

    categorical_steps = [
        ("imputer", SimpleImputer(strategy=preprocessing_config["categorical_imputation"]))
    ]
    if preprocessing_config.get("encode_categorical_features", True):
        categorical_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore")))
    else:
        raise ValueError("encode_categorical_features must be true for this classification pipeline.")
    categorical_pipeline = Pipeline(categorical_steps)

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])
    X_train_processed = preprocessor.fit_transform(X_train_clean)
    X_test_processed = preprocessor.transform(X_test_clean)

    print("Training data processed:", X_train_processed.shape)
    print("Testing data processed:", X_test_processed.shape)
    print("Missing values handled.")
    print(f"Outliers clipped using training-set quantiles: {lower_q} and {upper_q}.")
    return X_train_processed, X_test_processed, preprocessor


def engineer_features(X):
    """Create additional features for the Adult dataset."""
    X = X.copy()
    X["capital_net"] = X["capital-gain"] - X["capital-loss"]
    X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16)
    return X


def assess_label_quality(df, results_dir, target_column):
    """Run initial label-quality checks and export review samples."""
    print("\n--- Label Quality Assessment ---")
    sample = df.sample(n=50, random_state=42)
    sample_path = results_dir / "label_review_sample.csv"
    sample.to_csv(sample_path, index=False)
    print(f"Random sample saved to: {sample_path}")

    features = df.drop(columns=[target_column])
    label_counts = df.groupby(
        list(features.columns), dropna=False, observed=True
    )[target_column].nunique()
    conflicting_groups = int((label_counts > 1).sum())
    print("Feature groups with conflicting target labels:", conflicting_groups)

    manual_sample = df.sample(n=50, random_state=123)
    manual_path = results_dir / "manual_label_review.csv"
    manual_sample.to_csv(manual_path, index=False)
    print(f"Manual review sample saved to: {manual_path}")

    summary = {
        "random_sample_size": len(sample),
        "manual_review_sample_size": len(manual_sample),
        "conflicting_feature_groups": conflicting_groups,
        "label_column": target_column,
        "review_status": "Manual verification required",
    }
    summary_path = results_dir / "label_quality_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)
    print(f"Label quality summary saved to: {summary_path}")


def calculate_sha256(file_path):
    """Calculate the SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    config = load_config()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    target_column = config["dataset"]["target_column"]

    # Step 1: Load and validate dataset
    df = load_dataset(config)
    validate_dataset_structure(df, config)
    validate_data(df, target_column)
    save_validation_report(df, results_dir / "validation_report.txt")
    describe_problem(df, target_column)
    assess_label_quality(df, results_dir, target_column)

    # Step 2: Split before cleaning
    X_train, X_test, y_train, y_test = split_dataset(df, config)

    # Step 3: Remove exact duplicates from training data only
    print("\n--- Duplicate Removal ---")
    train_data = X_train.copy()
    train_data[target_column] = y_train
    rows_before = len(train_data)
    train_data = train_data.drop_duplicates()
    duplicates_removed = rows_before - len(train_data)
    y_train = train_data[target_column]
    X_train = train_data.drop(columns=[target_column])
    print(f"Training rows before removal: {rows_before}")
    print(f"Duplicate rows removed: {duplicates_removed}")
    print(f"Training rows after removal: {len(X_train)}")
    print("Testing data was not modified by duplicate removal.")

    # Step 4: Baseline preprocessing
    X_train_processed, X_test_processed, preprocessor = preprocess_data(X_train, X_test, config)
    joblib.dump(preprocessor, results_dir / "preprocessor.joblib")
    print("Preprocessor saved successfully.")

    # Save processed training data and its hash
    processed_path = results_dir / "processed_dataset.csv"
    processed_df = X_train.copy()
    processed_df[target_column] = y_train
    processed_df.to_csv(processed_path, index=False)
    dataset_hash = calculate_sha256(processed_path)
    with open(results_dir / "dataset_hash.txt", "w", encoding="utf-8") as file:
        file.write(f"File: {processed_path.name}\nSHA-256: {dataset_hash}\n")
    print(f"Processed dataset saved to: {processed_path}")
    print(f"SHA-256: {dataset_hash}")

    # Step 5: Feature engineering and comparison
    feature_config = config["feature_engineering"]
    model_config = config["model"]
    evaluation_config = config["evaluation"]

    baseline_model = LogisticRegression(
        max_iter=model_config["max_iter"],
        random_state=model_config["random_state"],
    )
    baseline_model.fit(X_train_processed, y_train)
    baseline_predictions = baseline_model.predict(X_test_processed)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    positive_label = evaluation_config["positive_label"]
    baseline_f1 = f1_score(y_test, baseline_predictions, pos_label=positive_label)
    print("\n--- Baseline Model ---")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    print(f"Baseline F1 score: {baseline_f1:.4f}")

    metrics = {
        "baseline_accuracy": float(baseline_accuracy),
        "baseline_f1_score": float(baseline_f1),
        "training_rows_after_duplicate_removal": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "training_duplicates_removed": int(duplicates_removed),
        "sha256_processed_dataset": dataset_hash,
    }

    if feature_config.get("enabled", True):
        X_train_engineered = engineer_features(X_train)
        X_test_engineered = engineer_features(X_test)
        print("\n--- Feature Engineering ---")
        print("New features added: capital_net, hours_per_age")

        X_train_engineered_processed, X_test_engineered_processed, engineered_preprocessor = preprocess_data(
            X_train_engineered, X_test_engineered, config
        )
        joblib.dump(engineered_preprocessor, results_dir / "engineered_preprocessor.joblib")
        print("Engineered preprocessor saved successfully.")

        engineered_model = LogisticRegression(
            max_iter=model_config["max_iter"],
            random_state=model_config["random_state"],
        )
        engineered_model.fit(X_train_engineered_processed, y_train)
        engineered_predictions = engineered_model.predict(X_test_engineered_processed)
        engineered_accuracy = accuracy_score(y_test, engineered_predictions)
        engineered_f1 = f1_score(y_test, engineered_predictions, pos_label=positive_label)

        print("\n--- Model With Engineered Features ---")
        print(f"Engineered accuracy: {engineered_accuracy:.4f}")
        print(f"Engineered F1 score: {engineered_f1:.4f}")
        print("\n--- Model Comparison ---")
        print(f"Accuracy change: {engineered_accuracy - baseline_accuracy:+.4f}")
        print(f"F1 score change: {engineered_f1 - baseline_f1:+.4f}")

        metrics.update({
            "engineered_accuracy": float(engineered_accuracy),
            "engineered_f1_score": float(engineered_f1),
            "accuracy_change": float(engineered_accuracy - baseline_accuracy),
            "f1_score_change": float(engineered_f1 - baseline_f1),
        })

    with open(results_dir / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    print(f"Model metrics saved to: {results_dir / 'model_metrics.json'}")
    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
