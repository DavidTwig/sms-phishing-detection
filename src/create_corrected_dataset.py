"""
Create Corrected SmishX Dataset
Relabels the 74 messages identified in the manual audit as mislabelled.
These messages are labelled 'smishing' in the original dataset but are
actually spam (no impersonation of a trusted entity).

Usage: python src/create_corrected_dataset.py

Input:  data/dataset.csv (original SmishX dataset)
        outputs/evaluation_no_context.json (evaluation results to identify the 74 messages)
Output: data/dataset_corrected.csv
"""

import pandas as pd
import json
import os
import sys

DATASET_PATH = "data/dataset.csv"
RESULTS_PATH = "outputs/evaluation_no_context.json"
OUTPUT_PATH = "data/dataset_corrected.csv"

# ============================================================
# Messages that ARE genuine smishing (keep as smishing)
# These are the 5 messages from the manual audit that involve
# real impersonation of companies (fake job offers).
# We identify them by substring matching.
# ============================================================
GENUINE_SMISHING_PATTERNS = [
    "Michael Hendrix from OPN Architects",
    "Your CV has passed",
    "Pruitt Group Business Account",
    "We found your contact on Indeed employment resume",
    "We need extra hands in our team",
    "you passed the interview"
]


def is_genuine_smishing(sms_text):
    """Check if a smishing->spam message is one of the 5 genuine smishing cases."""
    text_lower = str(sms_text).lower()
    for pattern in GENUINE_SMISHING_PATTERNS:
        if pattern.lower() in text_lower:
            return True
    return False


def main():
    # Load dataset
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    df['label_clean'] = df['label'].str.strip().str.lower()
    print(f"Loaded dataset: {len(df)} messages")
    print(f"Original distribution:\n{df['label_clean'].value_counts()}\n")

    # Load predictions to identify which smishing messages were predicted as spam
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: Results not found at '{RESULTS_PATH}'")
        print("Run the full evaluation first: python src/evaluate.py --no-context")
        sys.exit(1)

    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = results['predictions']

    if len(predictions) != len(df):
        print(f"ERROR: Prediction count ({len(predictions)}) doesn't match dataset ({len(df)})")
        sys.exit(1)

    df['predicted'] = predictions

    # Find smishing messages predicted as spam
    smishing_as_spam = df[
        (df['label_clean'] == 'smishing') &
        (df['predicted'] == 'spam')
    ].copy()

    print(f"Found {len(smishing_as_spam)} smishing messages predicted as spam")

    # Filter out the genuine smishing cases (keep those as smishing)
    to_relabel = []
    kept_as_smishing = []

    for idx, row in smishing_as_spam.iterrows():
        if is_genuine_smishing(row['SMS']):
            kept_as_smishing.append(idx)
        else:
            to_relabel.append(idx)

    print(f"Genuine smishing (keeping label): {len(kept_as_smishing)}")
    print(f"Mislabelled as smishing (relabelling to spam): {len(to_relabel)}")

    # Create corrected dataset
    df_corrected = df.copy()
    df_corrected.loc[to_relabel, 'label'] = 'spam'

    # Clean up - remove helper columns
    df_corrected = df_corrected.drop(columns=['label_clean', 'predicted'])

    # Verify new distribution
    df_corrected['label_check'] = df_corrected['label'].str.strip().str.lower()
    print(f"\nCorrected distribution:\n{df_corrected['label_check'].value_counts()}")
    df_corrected = df_corrected.drop(columns=['label_check'])

    # Save
    df_corrected.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"\nSaved corrected dataset to: {OUTPUT_PATH}")
    print(f"  Smishing count: {len(df[df['label_clean'] == 'smishing'])} -> {len(df_corrected[df_corrected['label'].str.strip().str.lower() == 'smishing'])}")
    print(f"  Spam count: {len(df[df['label_clean'] == 'spam'])} -> {len(df_corrected[df_corrected['label'].str.strip().str.lower() == 'spam'])}")
    print(f"  Legitimate count unchanged: {len(df[df['label_clean'] == 'legitimate'])}")

    # Show a few examples of relabelled messages
    print(f"\nSample relabelled messages (first 5):")
    for i, idx in enumerate(to_relabel[:5]):
        sms_preview = str(df.loc[idx, 'SMS'])[:100] + '...'
        print(f"  [{i+1}] {sms_preview}")


if __name__ == "__main__":
    main()