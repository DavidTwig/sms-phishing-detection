"""
Quick test script for SMS Phishing Detector.
Randomly samples messages from the dataset for testing.
"""

import pandas as pd
import argparse
from detector import SMSPhishingDetector
import urllib3

# Suppress SSL warnings for context gathering
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_sample(csv_path: str, n: int = 10, balanced: bool = True) -> pd.DataFrame:
    """
    Load a sample of messages from the dataset.
    
    Args:
        csv_path: Path to the dataset CSV
        n: Number of messages to sample
        balanced: If True, sample equally from each class
    
    Returns:
        DataFrame with sampled messages
    """
    df = pd.read_csv(csv_path)
    
    if balanced:
        # Sample equally from each class
        per_class = n // 3
        remainder = n % 3
        
        samples = []
        for i, label in enumerate(['legitimate', 'spam', 'smishing']):
            class_df = df[df['label'] == label]
            # Add remainder to first classes
            count = per_class + (1 if i < remainder else 0)
            count = min(count, len(class_df))  # Don't exceed available
            samples.append(class_df.sample(n=count))
        
        return pd.concat(samples).sample(frac=1)  # Shuffle
    else:
        return df.sample(n=min(n, len(df)))


def run_test(n: int = 10, use_context: bool = True, balanced: bool = True):
    """
    Run detector test on sampled messages.
    
    Args:
        n: Number of messages to test
        use_context: Whether to use context gathering
        balanced: Whether to balance across classes
    """
    print("=" * 70)
    print(f"SMS PHISHING DETECTOR TEST")
    print(f"Messages: {n} | Context gathering: {'ON' if use_context else 'OFF'} | Balanced: {balanced}")
    print("=" * 70)
    
    # Load sample
    print("\nLoading sample from dataset...")
    df = load_sample('data/dataset.csv', n=n, balanced=balanced)
    print(f"Loaded {len(df)} messages")
    
    # Initialise detector
    print(f"\nInitialising detector (context={'ON' if use_context else 'OFF'})...")
    detector = SMSPhishingDetector(use_context=use_context)
    
    # Track results
    correct = 0
    total = 0
    results = []
    
    # Test each message
    for idx, row in df.iterrows():
        total += 1
        sms = row['SMS']
        actual = row['label']
        
        print(f"\n{'='*70}")
        print(f"[{total}/{len(df)}] ACTUAL: {actual.upper()}")
        print(f"SMS: {sms[:100]}{'...' if len(sms) > 100 else ''}")
        print("-" * 70)
        
        # Run detection
        result = detector.detect(sms)
        predicted = result['classification']
        confidence = result['confidence']
        explanation = result['explanation']
        
        # Check if correct
        is_correct = predicted == actual
        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            status = "✗ WRONG"
        
        print(f"PREDICTED: {predicted.upper()} ({confidence}% confidence)")
        print(f"RESULT: {status}")
        print(f"EXPLANATION: {explanation}")
        
        # Show context if URLs were found
        if use_context and 'context' in result and result['context']['urls_found']:
            print(f"\nURLs analysed: {result['context']['urls_found']}")
            for analysis in result['context']['url_analyses']:
                if analysis['whois'].get('domain_age_days') is not None:
                    age = analysis['whois']['domain_age_days']
                    print(f"  Domain age: {age} days")
                if analysis['html'].get('password_field'):
                    print(f"  ⚠ Password field detected!")
        
        results.append({
            'actual': actual,
            'predicted': predicted,
            'correct': is_correct,
            'confidence': confidence
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    
    # Per-class breakdown
    for label in ['legitimate', 'spam', 'smishing']:
        class_results = [r for r in results if r['actual'] == label]
        if class_results:
            class_correct = sum(1 for r in class_results if r['correct'])
            print(f"  {label.capitalize()}: {class_correct}/{len(class_results)}")
    
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test SMS Phishing Detector')
    parser.add_argument('-n', type=int, default=10, help='Number of messages to test (default: 10)')
    parser.add_argument('--no-context', action='store_true', help='Disable context gathering')
    parser.add_argument('--unbalanced', action='store_true', help='Random sample instead of balanced')
    
    args = parser.parse_args()
    
    run_test(
        n=args.n,
        use_context=not args.no_context,
        balanced=not args.unbalanced
    )