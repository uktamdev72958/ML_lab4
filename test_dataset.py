from sklearn.datasets import fetch_openml

print("Downloading Adult dataset...")

adult = fetch_openml(
    name="adult",
    version=2,
    as_frame=True
)

df = adult.frame

print("Dataset downloaded successfully!")
print("Shape:", df.shape)
print(df.head())
print(df.columns.tolist())