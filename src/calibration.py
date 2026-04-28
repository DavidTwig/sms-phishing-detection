"""
Confidence Calibration Analysis for SMS Phishing Detection
Computes Expected Calibration Error (ECE) and generates reliability diagrams.

Usage: python src/calibration.py
       python src/calibration.py --file outputs/evaluation_no_context.json
       python src/calibration.py --bins 15
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
from collections import Counter


# ============================================================
# LOAD DATA
# ============================================================
def load_results(path):
    """Load evaluation results with predictions, confidences, and true labels."""
    if not os.path.exists(path):
        print(f"ERROR: Results file not found at '{path}'")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    
    predictions = data['predictions']
    if 'confidences_raw' in data:
        confidences = data['confidences_raw']
        print(f"Using raw (pre-Platt) confidences for calibration analysis")
    else:
        confidences = data['confidences']
        print(f"WARNING: No confidences_raw field found")
    true_labels = data['true_labels']

    print(f"Loaded {len(predictions)} predictions from {path}")
    print(f"Confidence range: {min(confidences)} - {max(confidences)}")
    print(f"Mean confidence: {np.mean(confidences):.1f}")

    return predictions, confidences, true_labels


# ============================================================
# ECE CALCULATION
# ============================================================
# ECE (Expected Calibration Error) answers the question:
# "When the model says it's 90% confident, is it actually right 90% of the time?"
# A low ECE means confidence scores are trustworthy. A high ECE means
# the model is over- or under-confident compared to how often it's correct.
# ECE = 0 would be perfect calibration.

def compute_ece(predictions, confidences, true_labels, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures how well confidence scores reflect actual accuracy.
    A perfectly calibrated model has ECE = 0.
    """
    # Convert confidence from 0-100 scale to 0-1 scale for the calculation
    confs = np.array(confidences) / 100.0
    # Create an array of 1s and 0s — 1 if prediction was correct, 0 if not
    correct = np.array([p == t for p, t in zip(predictions, true_labels)], dtype=float)

    # Split the 0-1 confidence range into equal-width bins
    # e.g. 10 bins = [0-0.1, 0.1-0.2, ..., 0.9-1.0]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_data = []
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Last bin uses <= instead of < so confidence of exactly 1.0 isn't missed
        if i == n_bins - 1:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        n_in_bin = in_bin.sum()

        # Skip empty bins
        if n_in_bin == 0:
            bin_data.append({
                'bin_lower': bin_lower, 'bin_upper': bin_upper,
                'bin_mid': (bin_lower + bin_upper) / 2,
                'count': 0, 'accuracy': 0, 'avg_confidence': 0, 'gap': 0
            })
            continue

        # For each bin, compare the average confidence to the actual accuracy
        # e.g. if messages in the 0.8-0.9 bin have average confidence 0.85
        # but only 0.70 accuracy, the gap is 0.15 (model is overconfident)
        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confs[in_bin].mean()
        gap = abs(bin_accuracy - bin_confidence)

        # ECE = weighted average of all bin gaps
        # Each bin's gap is weighted by what fraction of total predictions it holds
        # So bins with more predictions count more towards the final ECE score
        ece += (n_in_bin / len(confs)) * gap

        bin_data.append({
            'bin_lower': bin_lower, 'bin_upper': bin_upper,
            'bin_mid': (bin_lower + bin_upper) / 2,
            'count': int(n_in_bin), 'accuracy': float(bin_accuracy),
            'avg_confidence': float(bin_confidence), 'gap': float(gap)
        })

    return ece, bin_data


def compute_per_class_ece(predictions, confidences, true_labels, n_bins=10):
    """Compute ECE separately for each class."""
    classes = sorted(set(true_labels))
    results = {}

    for cls in classes:
        # Filter to only predictions where the true label is this class
        mask = [t == cls for t in true_labels]
        cls_preds = [p for p, m in zip(predictions, mask) if m]
        cls_confs = [c for c, m in zip(confidences, mask) if m]
        cls_true = [t for t, m in zip(true_labels, mask) if m]

        if len(cls_preds) == 0:
            continue

        # Reuse the same ECE function but only on this class's predictions
        ece, bin_data = compute_ece(cls_preds, cls_confs, cls_true, n_bins)
        accuracy = sum(p == t for p, t in zip(cls_preds, cls_true)) / len(cls_preds)

        results[cls] = {
            'ece': ece, 'accuracy': accuracy, 'count': len(cls_preds),
            'mean_confidence': np.mean(cls_confs), 'bin_data': bin_data
        }

    return results


# ============================================================
# RELIABILITY DIAGRAM
# ============================================================
# A reliability diagram is a bar chart where each bar represents a
# confidence bin. The bar height shows actual accuracy for that bin.
# If the model is well-calibrated, bars should follow the diagonal
# line (i.e. 80% confidence bin should have ~80% accuracy).

def plot_reliability_diagram(bin_data, ece, title="Reliability Diagram", save_path=None):
    """Plot a reliability diagram showing calibration."""
    # Two side-by-side plots: left = reliability diagram, right = bin counts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bin_mids = [b['bin_mid'] for b in bin_data]
    accuracies = [b['accuracy'] for b in bin_data]
    counts = [b['count'] for b in bin_data]
    width = bin_data[0]['bin_upper'] - bin_data[0]['bin_lower'] if bin_data else 0.1

    # Left plot: accuracy bars for each confidence bin
    bars = ax1.bar(bin_mids, accuracies, width=width * 0.85, alpha=0.7,
                   color='#4C72B0', edgecolor='#2E4057', linewidth=0.8, label='Accuracy')

    # Overlay red shading to show the "gap" between confidence and accuracy
    # — this is the calibration error for each bin, visually
    for i, b in enumerate(bin_data):
        if b['count'] > 0:
            gap_bottom = min(b['accuracy'], b['avg_confidence'])
            gap_height = abs(b['accuracy'] - b['avg_confidence'])
            ax1.bar(b['bin_mid'], gap_height, bottom=gap_bottom,
                    width=width * 0.85, alpha=0.3, color='#C44E52', edgecolor='none')

    # Diagonal line = perfect calibration (confidence matches accuracy exactly)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax1.set_xlabel('Mean Predicted Confidence', fontsize=12)
    ax1.set_ylabel('Fraction Correct (Accuracy)', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}', fontsize=13)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right plot: how many predictions fell in each confidence bin
    # Useful for seeing if the model clusters predictions in certain ranges
    ax2.bar(bin_mids, counts, width=width * 0.85, alpha=0.7,
            color='#55A868', edgecolor='#2E4057', linewidth=0.8)
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Number of Predictions', fontsize=12)
    ax2.set_title('Predictions per Confidence Bin', fontsize=13)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_per_class_reliability(per_class_results, save_path=None):
    """Plot reliability diagrams for each class side by side."""
    classes = list(per_class_results.keys())
    n_classes = len(classes)

    # One subplot per class (legitimate, spam, smishing)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
    if n_classes == 1:
        axes = [axes]

    # Different colour per class for visual distinction
    colors = {'legitimate': '#4C72B0', 'spam': '#55A868', 'smishing': '#C44E52'}

    for ax, cls in zip(axes, classes):
        data = per_class_results[cls]
        bin_data = data['bin_data']
        bin_mids = [b['bin_mid'] for b in bin_data]
        accuracies = [b['accuracy'] for b in bin_data]
        width = bin_data[0]['bin_upper'] - bin_data[0]['bin_lower'] if bin_data else 0.1
        color = colors.get(cls, '#4C72B0')

        ax.bar(bin_mids, accuracies, width=width * 0.85, alpha=0.7,
               color=color, edgecolor='#2E4057', linewidth=0.8)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
        ax.set_xlabel('Mean Predicted Confidence', fontsize=11)
        ax.set_ylabel('Fraction Correct', fontsize=11)
        ax.set_title(f'{cls.capitalize()}\nECE={data["ece"]:.4f} | Acc={data["accuracy"]:.1%} | n={data["count"]}', fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Class Reliability Diagrams', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_confidence_distribution(predictions, confidences, true_labels, save_path=None):
    """Plot confidence score distributions for correct vs incorrect predictions."""
    # Split confidence scores into two groups: correct and incorrect predictions
    correct_confs = [c for p, c, t in zip(predictions, confidences, true_labels) if p == t]
    wrong_confs = [c for p, c, t in zip(predictions, confidences, true_labels) if p != t]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = range(0, 105, 5)
    # Overlapping histograms — green for correct, red for incorrect
    ax.hist(correct_confs, bins=bins, alpha=0.6, color='#55A868', label=f'Correct (n={len(correct_confs)})', edgecolor='white')
    ax.hist(wrong_confs, bins=bins, alpha=0.6, color='#C44E52', label=f'Incorrect (n={len(wrong_confs)})', edgecolor='white')
    # Dashed lines showing the average confidence for each group
    ax.axvline(np.mean(correct_confs), color='#2E7D32', linestyle='--', linewidth=2,
               label=f'Correct mean: {np.mean(correct_confs):.1f}')
    ax.axvline(np.mean(wrong_confs), color='#B71C1C', linestyle='--', linewidth=2,
               label=f'Incorrect mean: {np.mean(wrong_confs):.1f}')
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Confidence Distribution: Correct vs Incorrect Predictions', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# PER-CONFUSION-TYPE CONFIDENCE ANALYSIS
# ============================================================
def analyse_confidence_by_error_type(predictions, confidences, true_labels):
    """Analyse confidence scores broken down by each confusion type."""
    print("\n" + "=" * 70)
    print("CONFIDENCE BY ERROR TYPE")
    print("=" * 70)

    # Group errors by confusion type (e.g. "legitimate -> spam", "spam -> smishing")
    error_groups = {}
    for p, c, t in zip(predictions, confidences, true_labels):
        if p != t:
            key = f"{t} -> {p}"
            if key not in error_groups:
                error_groups[key] = []
            error_groups[key].append(c)

    # Also collect confidence scores for correct predictions per class
    correct_groups = {}
    for p, c, t in zip(predictions, confidences, true_labels):
        if p == t:
            if t not in correct_groups:
                correct_groups[t] = []
            correct_groups[t].append(c)

    print("\nCorrect prediction confidence by class:")
    for cls in sorted(correct_groups.keys()):
        confs = correct_groups[cls]
        print(f"  {cls}: mean={np.mean(confs):.1f}, median={np.median(confs):.1f}, "
              f"min={min(confs)}, max={max(confs)}, n={len(confs)}")

    # Sort error types by frequency (most common first)
    print("\nError confidence by confusion type:")
    for key in sorted(error_groups.keys(), key=lambda k: -len(error_groups[k])):
        confs = error_groups[key]
        print(f"  {key}: mean={np.mean(confs):.1f}, median={np.median(confs):.1f}, "
              f"min={min(confs)}, max={max(confs)}, n={len(confs)}")
        # Check how many errors had low confidence — if most errors are
        # low-confidence, that means the model "knows" it's unsure
        for threshold in [60, 70, 80]:
            below = sum(1 for c in confs if c < threshold)
            print(f"    Below {threshold}%: {below}/{len(confs)} ({100*below/len(confs):.0f}%)")


# ============================================================
# TEXT REPORT
# ============================================================
def print_calibration_report(ece, bin_data, per_class_results, predictions, confidences, true_labels):
    """Print a detailed calibration analysis report."""
    print("=" * 70)
    print("CONFIDENCE CALIBRATION ANALYSIS")
    print("=" * 70)

    # Interpret the ECE score using standard thresholds
    print(f"\nOverall ECE: {ece:.4f}")
    if ece < 0.05:
        print("Interpretation: WELL CALIBRATED (ECE < 0.05)")
    elif ece < 0.10:
        print("Interpretation: MODERATELY CALIBRATED (0.05 <= ECE < 0.10)")
    elif ece < 0.20:
        print("Interpretation: POORLY CALIBRATED (0.10 <= ECE < 0.20)")
    else:
        print("Interpretation: VERY POORLY CALIBRATED (ECE >= 0.20)")

    correct_confs = [c for p, c, t in zip(predictions, confidences, true_labels) if p == t]
    wrong_confs = [c for p, c, t in zip(predictions, confidences, true_labels) if p != t]

    # Compare average confidence when right vs when wrong
    # Ideally the model should be less confident when it's wrong
    print(f"\nConfidence Statistics:")
    print(f"  Overall mean confidence: {np.mean(confidences):.1f}")
    print(f"  Correct predictions mean: {np.mean(correct_confs):.1f} (n={len(correct_confs)})")
    print(f"  Incorrect predictions mean: {np.mean(wrong_confs):.1f} (n={len(wrong_confs)})")
    print(f"  Confidence gap: {np.mean(correct_confs) - np.mean(wrong_confs):.1f} points")

    if np.mean(correct_confs) - np.mean(wrong_confs) > 10:
        print("  -> Good: model is notably less confident when wrong")
    elif np.mean(correct_confs) - np.mean(wrong_confs) > 0:
        print("  -> Weak signal: model is slightly less confident when wrong")
    else:
        print("  -> Warning: model is equally or more confident when wrong")

    print(f"\nPer-Bin Breakdown ({len(bin_data)} bins):")
    print(f"  {'Bin':<15} {'Count':<8} {'Accuracy':<12} {'Avg Conf':<12} {'Gap':<10}")
    print(f"  {'-'*55}")
    for b in bin_data:
        if b['count'] > 0:
            bin_label = f"{b['bin_lower']:.1f}-{b['bin_upper']:.1f}"
            # "over" = model is more confident than accurate (overconfident)
            # "under" = model is less confident than accurate (underconfident)
            direction = "over" if b['avg_confidence'] > b['accuracy'] else "under"
            print(f"  {bin_label:<15} {b['count']:<8} {b['accuracy']:<12.3f} {b['avg_confidence']:<12.3f} {b['gap']:<8.3f} ({direction})")

    print(f"\nPer-Class ECE:")
    print(f"  {'Class':<15} {'ECE':<10} {'Accuracy':<12} {'Mean Conf':<12} {'Count':<8}")
    print(f"  {'-'*55}")
    for cls, data in per_class_results.items():
        print(f"  {cls:<15} {data['ece']:<10.4f} {data['accuracy']:<12.1%} {data['mean_confidence']:<12.1f} {data['count']:<8}")

    # Find predictions where the model was very confident (>=80%) but wrong
    # These are the most dangerous errors — high confidence but incorrect
    high_conf_errors = [(p, c, t) for p, c, t in zip(predictions, confidences, true_labels)
                        if p != t and c >= 80]
    print(f"\nHigh-Confidence Errors (>=80% confidence but wrong): {len(high_conf_errors)}")
    if high_conf_errors:
        error_types = Counter(f"{t}->{p}" for p, c, t in high_conf_errors)
        for error_type, count in error_types.most_common():
            print(f"  {error_type}: {count}")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Confidence Calibration Analysis')
    parser.add_argument('--file', type=str, default='outputs/evaluation_no_context.json',
                        help='Path to evaluation results JSON')
    parser.add_argument('--bins', type=int, default=10,
                        help='Number of bins for ECE calculation (default: 10)')
    args = parser.parse_args()

    # Load the evaluation results (predictions, confidence scores, true labels)
    predictions, confidences, true_labels = load_results(args.file)

    # Compute overall ECE and per-class ECE
    ece, bin_data = compute_ece(predictions, confidences, true_labels, n_bins=args.bins)
    per_class_results = compute_per_class_ece(predictions, confidences, true_labels, n_bins=args.bins)

    # Print the text report and error analysis
    print_calibration_report(ece, bin_data, per_class_results, predictions, confidences, true_labels)
    analyse_confidence_by_error_type(predictions, confidences, true_labels)

    # Generate and save all plots
    os.makedirs('outputs', exist_ok=True)

    plot_reliability_diagram(bin_data, ece,
                            title="SMS Phishing Detector - Reliability Diagram",
                            save_path='outputs/reliability_diagram.png')
    plot_per_class_reliability(per_class_results,
                              save_path='outputs/reliability_per_class.png')
    plot_confidence_distribution(predictions, confidences, true_labels,
                                save_path='outputs/confidence_distribution.png')

    print(f"\nGenerated plots:")
    print(f"  outputs/reliability_diagram.png")
    print(f"  outputs/reliability_per_class.png")
    print(f"  outputs/confidence_distribution.png")


if __name__ == "__main__":
    main()