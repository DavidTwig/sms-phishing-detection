"""
Extract relabelled messages for manual verification.
Shows all messages that were changed from smishing to spam
in the corrected dataset.

Usage: python src/verify_relabelled.py
"""

import pandas as pd
import json

DATASET_PATH = "data/dataset.csv"
RESULTS_PATH = "outputs/evaluation_no_context.json"

# Genuine smishing patterns (keep as smishing)
GENUINE_SMISHING_PATTERNS = [
    "Michael Hendrix from OPN Architects",
    "Your CV has passed",
    "Pruitt Group Business Account",
    "We found your contact on Indeed employment resume",
    "We need extra hands in our team",
    "you passed the interview"
]

def is_genuine_smishing(sms_text):
    text_lower = str(sms_text).lower()
    for pattern in GENUINE_SMISHING_PATTERNS:
        if pattern.lower() in text_lower:
            return True
    return False

# Load data
df = pd.read_csv(DATASET_PATH)
df['label_clean'] = df['label'].str.strip().str.lower()

with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
    results = json.load(f)

df['predicted'] = results['predictions']

# Find smishing predicted as spam
smishing_as_spam = df[
    (df['label_clean'] == 'smishing') &
    (df['predicted'] == 'spam')
]

# Split into relabelled vs kept
relabelled = []
kept = []
for idx, row in smishing_as_spam.iterrows():
    if is_genuine_smishing(row['SMS']):
        kept.append(row)
    else:
        relabelled.append(row)

print(f"Total smishing->spam: {len(smishing_as_spam)}")
print(f"Kept as smishing: {len(kept)}")
print(f"Relabelled to spam: {len(relabelled)}")
print("=" * 70)

for i, row in enumerate(relabelled, 1):
    sms = str(row['SMS'])
    print(f"[{i}/{len(relabelled)}]")
    print(f"SMS: {sms}")
    print(f"VERDICT: _____ (smishing / spam)")
    print("-" * 70)