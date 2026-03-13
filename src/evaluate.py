"""
Evaluate the SMS Phishing Detector on the SmishX dataset.
Calculates accuracy, precision, recall, and F1 score.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from detector import SMSPhishingDetector
import time
import json
from datetime import datetime

def load_dataset(path: str) -> pd.DataFrame:
    """Load and clean the SmishX dataset."""
    df = pd.read_csv(path)
    
    # Clean the label column
    df['label'] = df['label'].str.strip().str.lower()
    
    return df

def evaluate_detector(detector: SMSPhishingDetector, df: pd.DataFrame, sample_size: int = None) -> dict:
    """
    Run the detector on the dataset and calculate metrics.
    
    Args:
        detector: The SMS phishing detector
        df: DataFrame with 'SMS' and 'label' columns
        sample_size: If set, only evaluate this many messages (for testing)
    
    Returns:
        Dictionary with predictions and metrics
    """
    
    if sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"Evaluating {len(df)} messages...")
    print("-" * 50)
    
    predictions = []
    confidences = []
    true_labels = []
    
    for idx, row in df.iterrows():
        sms = row['SMS']
        true_label = row['label']
        
        try:
            result = detector.detect(sms)
            pred_label = result['classification']
            confidence = result['confidence']
        except Exception as e:
            print(f"Error on message {idx}: {e}")
            pred_label = 'unknown'
            confidence = 0
        
        predictions.append(pred_label)
        confidences.append(confidence)
        true_labels.append(true_label)
        
        # Progress update every 10 messages
        if len(predictions) % 10 == 0:
            print(f"  Processed {len(predictions)}/{len(df)} messages...")
    
    print("-" * 50)
    print("Calculating metrics...")
    
    # Calculate metrics
    results = {
        'predictions': predictions,
        'confidences': confidences,
        'true_labels': true_labels,
        'accuracy': accuracy_score(true_labels, predictions),
        'report': classification_report(true_labels, predictions, output_dict=True),
        'confusion_matrix': confusion_matrix(true_labels, predictions, labels=['legitimate', 'spam', 'smishing']).tolist()
    }
    
    return results

def print_results(results: dict):
    """Print evaluation results in a readable format."""
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\nOverall Accuracy: {results['accuracy']:.1%}")
    
    print("\n" + "-" * 60)
    print("Per-Class Metrics:")
    print("-" * 60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    for label in ['legitimate', 'spam', 'smishing']:
        if label in results['report']:
            metrics = results['report'][label]
            print(f"{label:<15} {metrics['precision']:<12.3f} {metrics['recall']:<12.3f} {metrics['f1-score']:<12.3f} {metrics['support']:<10}")
    
    print("-" * 60)
    
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              legit   spam   smish")
    cm = results['confusion_matrix']
    labels = ['legitimate', 'spam', 'smishing']
    for i, label in enumerate(labels):
        print(f"Actual {label[:5]:>5}   {cm[i][0]:>5}  {cm[i][1]:>5}  {cm[i][2]:>5}")
    
    print("\n" + "=" * 60)

def save_results(results: dict, filename: str):
    """Save results to a JSON file."""
    
    # Convert to serialisable format
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': results['accuracy'],
        'report': results['report'],
        'confusion_matrix': results['confusion_matrix'],
        'predictions': results['predictions'],
        'confidences': results['confidences'],
        'true_labels': results['true_labels']
    }
    
    with open(filename, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    # Load dataset
    print("Loading SmishX dataset...")
    df = load_dataset('data/dataset.csv')
    print(f"Loaded {len(df)} messages")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Initialise detector
    print("\nInitialising detector...")
    detector = SMSPhishingDetector()
    
    # Run evaluation on a small sample first (to test)
    print("\n" + "=" * 60)
    print("RUNNING EVALUATION (all 1,200 messages)")
    print("=" * 60)
    
    results = evaluate_detector(detector, df, sample_size=None)
    print_results(results)
    
    # Save results
    save_results(results, 'outputs/evaluation_sample.json')
    
    print("\nTo run full evaluation, edit this file and change sample_size=20 to sample_size=None")
