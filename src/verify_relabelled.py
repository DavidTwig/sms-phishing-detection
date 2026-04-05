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

# These 5 messages were flagged as smishing->spam by the model but
# are actually genuine smishing (fake job offers impersonating companies).
# They get excluded from relabelling — same list as in
# create_corrected_dataset.py.
GENUINE_SMISHING_PATTERNS = [
    "Michael Hendrix from OPN Architects",
    "Your CV has passed",
    "Pruitt Group Business Account",
    "We found your contact on Indeed employment resume",
    "We need extra hands in our team",
    "you passed the interview"
]

# Check if a message matches one of the 5 genuine smishing patterns
def is_genuine_smishing(sms_text):
    text_lower = str(sms_text).lower()
    for pattern in GENUINE_SMISHING_PATTERNS:
        if pattern.lower() in text_lower:
            return True
    return False

# Load the original dataset and the model's predictions
df = pd.read_csv(DATASET_PATH)
df['label_clean'] = df['label'].str.strip().str.lower()

with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
    results = json.load(f)

df['predicted'] = results['predictions']

# Find messages labelled smishing but predicted as spam
# — these are the candidates that were relabelled in the corrected dataset
smishing_as_spam = df[
    (df['label_clean'] == 'smishing') &
    (df['predicted'] == 'spam')
]

# Separate into two groups:
# - relabelled: changed from smishing to spam in the corrected dataset
# - kept: the 5 genuine smishing messages that stay as smishing
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

# Print each relabelled message with a blank VERDICT line
# for manual review — same format as extract_smishing_spam.py
for i, row in enumerate(relabelled, 1):
    sms = str(row['SMS'])
    print(f"[{i}/{len(relabelled)}]")
    print(f"SMS: {sms}")
    print(f"VERDICT: _____ (smishing / spam)")
    print("-" * 70)