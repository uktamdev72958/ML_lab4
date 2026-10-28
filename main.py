from sklearn.datasets import fetch_openml


def load_dataset():
    """Load the Adult dataset from OpenML."""
    print("Loading the Adult dataset...")

    adult = fetch_openml(
        name="adult",
        version=2,
        as_frame=True
    )

    return adult.frame


def describe_problem(df):
    """Display the dataset and ML problem information."""
    print("\n--- Dataset Information ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Input features: {df.shape[1] - 1}")
    print("Target column: class")
    print("Problem type: Binary classification")
    print("Goal: Predict whether annual income exceeds $50,000.")


if __name__ == "__main__":
    df = load_dataset()
    describe_problem(df)