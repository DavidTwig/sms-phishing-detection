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
# ============================================================
# During the manual audit, 5 messages were found that the model
# predicted as spam but are actually real smishing — they involve
# fake job offers impersonating real companies.
# These are identified by substring matching so they don't get
# accidentally relabelled to spam along with the rest.
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
    # Convert to lowercase so matching isn't case-sensitive
    text_lower = str(sms_text).lower()
    for pattern in GENUINE_SMISHING_PATTERNS:
        if pattern.lower() in text_lower:
            return True
    return False


def main():
    # Load the original SmishX dataset
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'")
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    # Create a cleaned version of labels for reliable comparison
    df['label_clean'] = df['label'].str.strip().str.lower()
    print(f"Loaded dataset: {len(df)} messages")
    print(f"Original distribution:\n{df['label_clean'].value_counts()}\n")

    # Load the model's predictions — needed to find which smishing
    # messages the model predicted as spam (the suspected mislabels)
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: Results not found at '{RESULTS_PATH}'")
        print("Run the full evaluation first: python src/evaluate.py --no-context")
        sys.exit(1)

    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        results = json.load(f)

    predictions = results['predictions']

    # Sanity check — predictions must line up 1:1 with dataset rows
    if len(predictions) != len(df):
        print(f"ERROR: Prediction count ({len(predictions)}) doesn't match dataset ({len(df)})")
        sys.exit(1)

    df['predicted'] = predictions

    # Find messages labelled smishing but predicted as spam
    # These are the candidates for relabelling
    smishing_as_spam = df[
        (df['label_clean'] == 'smishing') &
        (df['predicted'] == 'spam')
    ].copy()

    print(f"Found {len(smishing_as_spam)} smishing messages predicted as spam")

    # Split into two groups:
    # 1. Genuine smishing (the 5 fake job offer messages) — keep as smishing
    # 2. Everything else — relabel from smishing to spam
    to_relabel = []
    kept_as_smishing = []

    for idx, row in smishing_as_spam.iterrows():
        if is_genuine_smishing(row['SMS']):
            kept_as_smishing.append(idx)
        else:
            to_relabel.append(idx)

    print(f"Genuine smishing (keeping label): {len(kept_as_smishing)}")
    print(f"Mislabelled as smishing (relabelling to spam): {len(to_relabel)}")

    # Apply the relabelling — change "smishing" to "spam" for the identified rows
    df_corrected = df.copy()
    df_corrected.loc[to_relabel, 'label'] = 'spam'

    # Remove the temporary columns that were only needed for processing
    df_corrected = df_corrected.drop(columns=['label_clean', 'predicted'])

    # Print the new distribution to confirm the changes look right
    df_corrected['label_check'] = df_corrected['label'].str.strip().str.lower()
    print(f"\nCorrected distribution:\n{df_corrected['label_check'].value_counts()}")
    df_corrected = df_corrected.drop(columns=['label_check'])

    # Save the corrected dataset
    df_corrected.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"\nSaved corrected dataset to: {OUTPUT_PATH}")
    print(f"  Smishing count: {len(df[df['label_clean'] == 'smishing'])} -> {len(df_corrected[df_corrected['label'].str.strip().str.lower() == 'smishing'])}")
    print(f"  Spam count: {len(df[df['label_clean'] == 'spam'])} -> {len(df_corrected[df_corrected['label'].str.strip().str.lower() == 'spam'])}")
    print(f"  Legitimate count unchanged: {len(df[df['label_clean'] == 'legitimate'])}")

    # Print a few examples of relabelled messages for a quick sanity check
    print(f"\nSample relabelled messages (first 5):")
    for i, idx in enumerate(to_relabel[:5]):
        sms_preview = str(df.loc[idx, 'SMS'])[:100] + '...'
        print(f"  [{i+1}] {sms_preview}")


if __name__ == "__main__":
    main()