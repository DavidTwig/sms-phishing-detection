"""
Explore the SmishX dataset to understand its structure.
"""

import pandas as pd

# Load the dataset
df = pd.read_csv('data/dataset.csv')

# Basic info
print("=" * 50)
print("SMISHX DATASET OVERVIEW")
print("=" * 50)

print(f"\nTotal messages: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

# Check for label distribution
print("\n" + "=" * 50)
print("LABEL DISTRIBUTION")
print("=" * 50)

for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() < 10:
        print(f"\n{col}:")
        print(df[col].value_counts())
