
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pathlib import Path

# --------------------------------------------------
# 1. LOAD THE DATASET
# --------------------------------------------------

file_path = Path(__file__).parent / "titanic_train.csv"
df = pd.read_csv(file_path)

print("FIRST FIVE ROWS")
print(df.head())

# --------------------------------------------------
# 2. BASIC DATASET INFORMATION
# --------------------------------------------------

print("\nDATASET INFORMATION")
df.info()

print("\nBASIC STATISTICS")
print(df.describe())

# --------------------------------------------------
# 3. CHECK MISSING VALUES
# --------------------------------------------------

print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

# Fill missing Age values using the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Fare values using the mean
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

# Fill missing Embarked values using the mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Cabin has many missing values, so remove this column
if "Cabin" in df.columns:
    df.drop(columns=["Cabin"], inplace=True)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# --------------------------------------------------
# 5. ENCODE THE SEX COLUMN USING LABEL ENCODER
# --------------------------------------------------

label_encoder = LabelEncoder()

df["Sex"] = label_encoder.fit_transform(df["Sex"])

print("\nSEX COLUMN AFTER LABEL ENCODING")
print(df["Sex"].head())

# --------------------------------------------------
# 6. ENCODE EMBARKED USING ONE HOT ENCODER
# --------------------------------------------------

onehot_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

encoded_values = onehot_encoder.fit_transform(
    df[["Embarked"]]
)

encoded_columns = onehot_encoder.get_feature_names_out(
    ["Embarked"]
)

encoded_df = pd.DataFrame(
    encoded_values,
    columns=encoded_columns,
    index=df.index
)

# Replace the original Embarked column with encoded columns
df.drop(columns=["Embarked"], inplace=True)

df = pd.concat([df, encoded_df], axis=1)

print("\nDATA AFTER CATEGORICAL ENCODING")
print(df.head())

# --------------------------------------------------
# 7. VISUALIZE AGE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(df["Age"], bins=20, edgecolor="black")

plt.title("Titanic Passenger Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.tight_layout()
plt.savefig(
    Path(__file__).parent / "age_distribution.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# 8. SAVE THE CLEANED DATASET
# --------------------------------------------------

output_path = Path(__file__).parent / "titanic_cleaned.csv"

df.to_csv(output_path, index=False)

print("\nCLEANED DATASET SAVED SUCCESSFULLY!")
print(f"File location: {output_path}")

print("\nFINAL DATASET SHAPE")
print(df.shape)

print("\nWEEK 1 ASSIGNMENT COMPLETED!")
