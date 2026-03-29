"""
Error Analysis Script for SMS Phishing Detection
Examines misclassified messages from evaluation results to identify patterns.

Usage: python src/error_analysis.py
  (run from the project root: sms-phishing-detection/)

Expects: outputs/evaluation_sample.json (from evaluate.py)
         data/dataset.csv (original dataset)
"""

import json
import pandas as pd
import os
import sys
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
RESULTS_PATH = "outputs/evaluation_no_context.json"
DATASET_PATH = "data/dataset_corrected.csv"
OUTPUT_DIR = "outputs"

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    """Load evaluation results and dataset, merge them."""
    if not os.path.exists(RESULTS_PATH):
        print(f"ERROR: Results file not found at '{RESULTS_PATH}'")
        print("Make sure you've run evaluate.py first.")
        sys.exit(1)
    
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at '{DATASET_PATH}'")
        sys.exit(1)
    
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    df = pd.read_csv(DATASET_PATH)
    predictions = results['predictions']
    
    print(f"Dataset rows: {len(df)}")
    print(f"Predictions:  {len(predictions)}")
    
    if len(df) != len(predictions):
        print(f"WARNING: Length mismatch! Using first {min(len(df), len(predictions))} rows.")
        n = min(len(df), len(predictions))
        df = df.iloc[:n].copy()
        predictions = predictions[:n]
    
    df['predicted'] = predictions
    df['true_label'] = df['label'].str.lower().str.strip()
    df['predicted'] = df['predicted'].str.lower().str.strip()
    df['correct'] = df['true_label'] == df['predicted']
    df['confusion_type'] = df['true_label'] + ' -> ' + df['predicted']
    
    print(f"Overall accuracy: {df['correct'].mean():.1%}")
    print(f"Correct: {df['correct'].sum()} | Misclassified: {(~df['correct']).sum()}")
    
    return df, results


# ============================================================
# PATTERN ANALYSIS HELPERS
# ============================================================
def check_url_presence(text):
    """Check if message contains URLs."""
    url_indicators = ['http://', 'https://', 'www.', '.com', '.co.uk', '.org',
                      'bit.ly', 'tinyurl', '.net', '.info', '.xyz']
    text_lower = str(text).lower()
    return any(ind in text_lower for ind in url_indicators)


def check_urgency_language(text):
    """Check for urgency/pressure language common in phishing."""
    urgency_words = [
        'urgent', 'immediately', 'expire', 'suspend', 'verify', 'confirm',
        'account', 'locked', 'unauthori', 'unusual activity', 'click now',
        'within 24', 'action required', 'limited time', 'act now', 'asap',
        'warning', 'alert', 'security', 'blocked', 'disabled'
    ]
    text_lower = str(text).lower()
    return [w for w in urgency_words if w in text_lower]


def check_impersonation_signals(text):
    """Check for brand/entity impersonation signals."""
    brands = [
        'bank', 'paypal', 'amazon', 'apple', 'microsoft', 'netflix', 'royal mail',
        'hmrc', 'dvla', 'nhs', 'post office', 'hermes', 'dpd', 'fedex', 'ups',
        'dhl', 'usps', 'irs', 'gov', 'customs', 'police', 'wells fargo',
        'chase', 'barclays', 'hsbc', 'lloyds', 'santander', 'halifax',
        'vodafone', 'ee', 'o2', 'bt', 'sky', 'argos', 'tesco', 'asda'
    ]
    text_lower = str(text).lower()
    return [b for b in brands if b in text_lower]


def check_money_language(text):
    """Check for financial/monetary language (more common in spam)."""
    money_words = [
        'win', 'prize', 'cash', 'free', 'claim', 'reward', 'offer',
        'discount', 'deal', 'bonus', 'credit', 'loan', 'invest',
        'bitcoin', 'crypto', 'gambling', 'bet', 'lottery', 'jackpot',
        '£', '$', 'pounds', 'dollars'
    ]
    text_lower = str(text).lower()
    return [w for w in money_words if w in text_lower]


def message_length(text):
    return len(str(text))


# ============================================================
# REPORTING
# ============================================================
def sep(char='=', length=70):
    print(char * length)


def analyse_confusion_group(df_errors, confusion_type, max_show=15):
    """Deep-dive into a specific confusion type."""
    group = df_errors[df_errors['confusion_type'] == confusion_type].copy()
    if len(group) == 0:
        return
    
    sep('-')
    print(f"DETAILED ANALYSIS: {confusion_type.upper()} ({len(group)} messages)")
    sep('-')
    
    # URL analysis
    group['has_url'] = group['SMS'].apply(check_url_presence)
    url_count = group['has_url'].sum()
    print(f"\n  Contains URL: {url_count}/{len(group)} ({100*url_count/len(group):.0f}%)")
    
    # Check dataset URL flag if available
    if 'if_URL' in group.columns:
        flagged = int(group['if_URL'].sum())
        print(f"  Dataset URL flag (if_URL=1): {flagged}/{len(group)}")
    
    # Urgency language
    group['urgency'] = group['SMS'].apply(check_urgency_language)
    urgency_count = group['urgency'].apply(len).gt(0).sum()
    print(f"  Has urgency language: {urgency_count}/{len(group)} ({100*urgency_count/len(group):.0f}%)")
    
    # Impersonation signals
    group['brands'] = group['SMS'].apply(check_impersonation_signals)
    brand_count = group['brands'].apply(len).gt(0).sum()
    print(f"  Has brand/entity names: {brand_count}/{len(group)} ({100*brand_count/len(group):.0f}%)")
    
    # Money language
    group['money'] = group['SMS'].apply(check_money_language)
    money_count = group['money'].apply(len).gt(0).sum()
    print(f"  Has money/prize language: {money_count}/{len(group)} ({100*money_count/len(group):.0f}%)")
    
    # Phone/email presence from dataset
    if 'if_phone' in group.columns:
        phone_count = int(group['if_phone'].sum())
        print(f"  Contains phone number: {phone_count}/{len(group)}")
    if 'if_email' in group.columns:
        email_count = int(group['if_email'].sum())
        print(f"  Contains email: {email_count}/{len(group)}")
    
    # Message length stats
    group['msg_len'] = group['SMS'].apply(message_length)
    print(f"\n  Message length: mean={group['msg_len'].mean():.0f}, "
          f"min={group['msg_len'].min()}, max={group['msg_len'].max()}")
    
    # Most common urgency words across the group
    all_urgency = []
    for words in group['urgency']:
        all_urgency.extend(words)
    if all_urgency:
        print(f"\n  Most common urgency words: {Counter(all_urgency).most_common(5)}")
    
    all_brands = []
    for brands in group['brands']:
        all_brands.extend(brands)
    if all_brands:
        print(f"  Most common brand mentions: {Counter(all_brands).most_common(5)}")
    
    all_money = []
    for words in group['money']:
        all_money.extend(words)
    if all_money:
        print(f"  Most common money words: {Counter(all_money).most_common(5)}")
    
    # Show individual messages
    print(f"\n  --- Sample messages ({min(max_show, len(group))} of {len(group)}) ---\n")
    for i, (_, row) in enumerate(group.head(max_show).iterrows()):
        sms_preview = str(row['SMS'])[:250] + ('...' if len(str(row['SMS'])) > 250 else '')
        print(f"  [{i+1}] TRUE: {row['true_label']} | PREDICTED: {row['predicted']}")
        print(f"      SMS: {sms_preview}")
        
        extras = []
        if row['urgency']:
            extras.append(f"Urgency: {row['urgency']}")
        if row['brands']:
            extras.append(f"Brands: {row['brands']}")
        if row['money']:
            extras.append(f"Money: {row['money']}")
        if extras:
            print(f"      Signals: {' | '.join(extras)}")
        print()


def summary_table(df):
    """Print a compact per-class accuracy table."""
    sep()
    print("PER-CLASS BREAKDOWN")
    sep()
    
    for label in ['legitimate', 'spam', 'smishing']:
        subset = df[df['true_label'] == label]
        correct = subset['correct'].sum()
        total = len(subset)
        
        # Where do the errors go?
        errors = subset[~subset['correct']]
        error_dest = Counter(errors['predicted'])
        error_str = ', '.join(f"{count}->{pred}" for pred, count in error_dest.most_common())
        
        print(f"\n  {label.upper()} ({total} total, {correct} correct = {100*correct/total:.1f}%)")
        if error_str:
            print(f"    Errors: {error_str}")


# ============================================================
# MAIN
# ============================================================
def main():
    sep()
    print("SMS PHISHING DETECTION - ERROR ANALYSIS")
    sep()
    
    df, results = load_data()
    
    # Per-class summary
    summary_table(df)
    
    # Confusion type summary
    misclassified = df[~df['correct']]
    sep()
    print(f"\nCONFUSION TYPE RANKING (total misclassified: {len(misclassified)})")
    sep()
    
    confusion_counts = Counter(misclassified['confusion_type'])
    for ctype, count in confusion_counts.most_common():
        pct_of_class = 100 * count / len(df[df['true_label'] == ctype.split(' -> ')[0]])
        print(f"  {ctype}: {count} messages ({pct_of_class:.1f}% of that class)")
    
    # Detailed analysis for the major confusion types
    for ctype, count in confusion_counts.most_common():
        if count >= 5:
            analyse_confusion_group(misclassified, ctype, max_show=15)
    
    # Save misclassified messages to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "error_analysis.csv")
    
    export_cols = ['SMS', 'true_label', 'predicted', 'confusion_type']
    if 'if_URL' in df.columns:
        export_cols.append('if_URL')
    if 'if_phone' in df.columns:
        export_cols.append('if_phone')
    if 'if_email' in df.columns:
        export_cols.append('if_email')
    
    misclassified[export_cols].sort_values('confusion_type').to_csv(
        csv_path, index=False, encoding='utf-8'
    )
    print(f"\n{'='*70}")
    print(f"Saved {len(misclassified)} misclassified messages to: {csv_path}")
    print(f"Open in Excel to review all errors manually.")
    print(f"\nTip: pipe output to file with:")
    print(f"  python src/error_analysis.py > outputs/error_analysis_report.txt")
    sep()


if __name__ == "__main__":
    main()