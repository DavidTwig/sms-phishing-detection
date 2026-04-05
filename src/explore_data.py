"""
Explore the SmishX dataset to understand its structure.
"""

import pandas as pd

# Load the original SmishX dataset (1,200 SMS messages)
df = pd.read_csv('data/dataset.csv')

# Print basic dataset info — row count, column names, sample rows, data types
print("=" * 50)
print("SMISHX DATASET OVERVIEW")
print("=" * 50)

print(f"\nTotal messages: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

# Show the value counts for any text column with fewer than 10 unique values
# In practice this picks up the 'label' column (legitimate/spam/smishing)
# and any other categorical flags in the dataset
print("\n" + "=" * 50)
print("LABEL DISTRIBUTION")
print("=" * 50)

for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() < 10:
        print(f"\n{col}:")
        print(df[col].value_counts())