"""
Extract smishing->spam misclassifications for manual audit.
Outputs a simple numbered list for easy review.

Usage: python src/extract_smishing_spam.py
"""

import json
import pandas as pd

RESULTS_PATH = "outputs/evaluation_sample.json"
DATASET_PATH = "data/dataset.csv"
OUTPUT_PATH = "outputs/smishing_as_spam_audit.txt"

# Load the evaluation predictions and the original dataset
with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
    results = json.load(f)

df = pd.read_csv(DATASET_PATH)
df['predicted'] = results['predictions']
df['true_label'] = df['label'].str.lower().str.strip()
df['predicted'] = df['predicted'].str.lower().str.strip()

# Find messages the dataset says are smishing but the model predicted as spam.
# These are the suspected mislabels — the manual audit checks whether the
# dataset or the model is correct for each one.
mask = (df['true_label'] == 'smishing') & (df['predicted'] == 'spam')
errors = df[mask].copy()

print(f"Found {len(errors)} smishing->spam misclassifications")

# Write a plain text file with each message and a blank VERDICT line
# to fill in manually. The key question for each message is whether the
# sender is impersonating a trusted entity (smishing) or just sending
# unsolicited promotional content (spam).
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(f"MANUAL AUDIT: {len(errors)} messages labelled SMISHING but predicted SPAM\n")
    f.write(f"Question for each: Is the sender IMPERSONATING a trusted entity?\n")
    f.write(f"  - YES = dataset is correct (genuine smishing)\n")
    f.write(f"  - NO  = model is correct (this is spam, not smishing)\n")
    f.write("=" * 70 + "\n\n")

    for i, (_, row) in enumerate(errors.iterrows(), 1):
        f.write(f"[{i}/{len(errors)}]\n")
        f.write(f"SMS: {row['SMS']}\n")
        f.write(f"Has URL: {'Yes' if row.get('if_URL', 0) == 1 else 'No'}\n")
        f.write(f"VERDICT: _____ (smishing / spam)\n")
        f.write("-" * 70 + "\n\n")

print(f"Saved to {OUTPUT_PATH}")
print(f"Open the file, read each message, and fill in the VERDICT line.")