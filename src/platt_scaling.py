"""
Platt Scaling for Confidence Calibration
Applies post-hoc calibration to the detector's confidence scores
using Platt scaling (logistic regression on raw confidence scores).

Based on: Platt, J. (2000). "Probabilistic Outputs for Support Vector
Machines and Comparisons to Regularized Likelihood Methods."

Usage: python src/platt_scaling.py
       python src/platt_scaling.py --file outputs/evaluation_no_context.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import argparse
import os
import sys
from collections import Counter


def load_results(path):
    """Load evaluation results."""
    if not os.path.exists(path):
        print(f"ERROR: Results file not found at '{path}'")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = data['predictions']
    confidences = data['confidences']
    true_labels = data['true_labels']

    print(f"Loaded {len(predictions)} predictions from {path}")
    return predictions, confidences, true_labels


# ============================================================
# ECE CALCULATION
# ============================================================
# Same ECE logic as calibration.py — groups predictions into bins
# by confidence level and measures the gap between confidence and
# actual accuracy in each bin. See calibration.py for a more
# detailed explanation.

def compute_ece(correct, confidences_norm, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Last bin uses <= so a confidence of exactly 1.0 isn't missed
        if i == n_bins - 1:
            in_bin = (confidences_norm >= bin_lower) & (confidences_norm <= bin_upper)
        else:
            in_bin = (confidences_norm >= bin_lower) & (confidences_norm < bin_upper)

        n_in_bin = in_bin.sum()
        if n_in_bin == 0:
            bin_data.append({'bin_mid': (bin_lower + bin_upper) / 2, 'count': 0,
                           'accuracy': 0, 'avg_confidence': 0})
            continue

        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confidences_norm[in_bin].mean()
        gap = abs(bin_accuracy - bin_confidence)
        # Weight each bin's gap by what fraction of predictions it holds
        ece += (n_in_bin / len(confidences_norm)) * gap

        bin_data.append({
            'bin_mid': (bin_lower + bin_upper) / 2,
            'bin_lower': bin_lower,
            'bin_upper': bin_upper,
            'count': int(n_in_bin),
            'accuracy': float(bin_accuracy),
            'avg_confidence': float(bin_confidence)
        })

    return ece, bin_data


# ============================================================
# PLATT SCALING — OVERALL
# ============================================================
# Platt scaling fits a logistic regression that learns the relationship
# between raw confidence and whether the prediction was actually correct.
# The fitted model then maps raw scores to calibrated probabilities.
#
# Cross-validation is used here (unlike fit_platt_params.py which fits
# on all data at once) so that every prediction gets a calibrated score
# without using data it was trained on — this avoids overfitting and
# gives a fairer estimate of how well calibration actually works.

def apply_platt_scaling(predictions, confidences, true_labels):
    """
    Apply Platt scaling using cross-validation to avoid overfitting.
    
    For each class, fits a logistic regression that maps raw confidence
    scores to calibrated probabilities. Uses 5-fold stratified CV so
    every prediction gets a calibrated score without data leakage.
    """
    confs = np.array(confidences) / 100.0
    correct = np.array([p == t for p, t in zip(predictions, true_labels)], dtype=int)

    calibrated_confs = np.zeros(len(confs))
    
    # StratifiedKFold ensures each fold has roughly the same ratio of
    # correct/incorrect predictions as the full dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, test_idx in skf.split(confs, correct):
        # Train the logistic regression on 4 folds
        train_confs = confs[train_idx].reshape(-1, 1)
        train_correct = correct[train_idx]
        
        lr = LogisticRegression(random_state=42)
        lr.fit(train_confs, train_correct)
        
        # Apply calibration to the held-out fold
        test_confs = confs[test_idx].reshape(-1, 1)
        # predict_proba returns [P(incorrect), P(correct)] — take P(correct)
        calibrated_probs = lr.predict_proba(test_confs)[:, 1]  # P(correct)
        calibrated_confs[test_idx] = calibrated_probs
    
    return calibrated_confs, correct


# ============================================================
# PLATT SCALING — PER-CLASS
# ============================================================
# Same idea as above but fits a separate logistic regression for each
# class (legitimate, spam, smishing). This is useful because different
# classes can have very different calibration issues — e.g. smishing
# predictions tend to be much more overconfident than spam predictions.

def apply_per_class_platt_scaling(predictions, confidences, true_labels):
    """
    Apply Platt scaling separately for each predicted class.
    This accounts for the fact that different classes may have
    different calibration curves (e.g., smishing is much more
    overconfident than spam).
    """
    confs = np.array(confidences) / 100.0
    correct = np.array([p == t for p, t in zip(predictions, true_labels)], dtype=int)
    classes = sorted(set(predictions))
    
    calibrated_confs = np.zeros(len(confs))
    
    for cls in classes:
        cls_mask = np.array([p == cls for p in predictions])
        cls_indices = np.where(cls_mask)[0]
        
        # Need enough samples to fit a meaningful logistic regression
        if len(cls_indices) < 10:
            calibrated_confs[cls_indices] = confs[cls_indices]
            continue
        
        cls_confs = confs[cls_indices].reshape(-1, 1)
        cls_correct = correct[cls_indices]
        
        # Logistic regression needs both correct and incorrect examples
        # to learn from — if all predictions for a class are correct
        # (or all wrong), there's nothing to calibrate
        if len(set(cls_correct)) < 2:
            calibrated_confs[cls_indices] = confs[cls_indices]
            continue
        
        # Number of CV folds can't exceed the smallest group size
        # e.g. if only 3 incorrect predictions, max 3 folds
        n_splits = min(5, min(Counter(cls_correct).values()))
        if n_splits < 2:
            calibrated_confs[cls_indices] = confs[cls_indices]
            continue
            
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(cls_confs, cls_correct):
            lr = LogisticRegression(random_state=42)
            lr.fit(cls_confs[train_idx], cls_correct[train_idx])
            calibrated_probs = lr.predict_proba(cls_confs[test_idx])[:, 1]
            
            # Map back from within-class indices to full dataset indices
            actual_indices = cls_indices[test_idx]
            calibrated_confs[actual_indices] = calibrated_probs
    
    return calibrated_confs, correct


# ============================================================
# PLOTTING
# ============================================================

def plot_before_after(bin_data_before, ece_before, bin_data_after, ece_after, 
                      title="Platt Scaling: Before vs After", save_path=None):
    """Plot reliability diagrams before and after calibration side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, bin_data, ece, subtitle in [
        (ax1, bin_data_before, ece_before, "Before (Raw Confidence)"),
        (ax2, bin_data_after, ece_after, "After (Platt Scaled)")
    ]:
        bin_mids = [b['bin_mid'] for b in bin_data]
        accuracies = [b['accuracy'] for b in bin_data]
        counts = [b['count'] for b in bin_data]
        
        width = 0.1
        
        bars = ax.bar(bin_mids, accuracies, width=width * 0.85, alpha=0.7,
                      color='#4C72B0', edgecolor='#2E4057', linewidth=0.8, label='Accuracy')
        
        # Red shading shows the calibration gap for each bin
        for b in bin_data:
            if b['count'] > 0:
                gap_bottom = min(b['accuracy'], b['avg_confidence'])
                gap_height = abs(b['accuracy'] - b['avg_confidence'])
                ax.bar(b['bin_mid'], gap_height, bottom=gap_bottom,
                       width=width * 0.85, alpha=0.3, color='#C44E52', edgecolor='none')
        
        # Diagonal = perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Confidence', fontsize=11)
        ax.set_ylabel('Fraction Correct (Accuracy)', fontsize=11)
        ax.set_title(f'{subtitle}\nECE = {ece:.4f}', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Show how many predictions are in each bin (below the bars)
        for b in bin_data:
            if b['count'] > 0:
                ax.text(b['bin_mid'], -0.08, str(b['count']), ha='center', fontsize=8, color='gray')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_confidence_shift(original_confs, calibrated_confs, correct, save_path=None):
    """Plot how confidence scores shifted after calibration."""
    # Each dot is one prediction — x-axis is the original score,
    # y-axis is the calibrated score. Dots on the diagonal didn't change.
    fig, ax = plt.subplots(figsize=(10, 5))
    
    correct_mask = correct == 1
    # Green dots = correct predictions, red dots = incorrect
    ax.scatter(original_confs[correct_mask], calibrated_confs[correct_mask],
              alpha=0.3, color='#55A868', label=f'Correct (n={correct_mask.sum()})', s=20)
    ax.scatter(original_confs[~correct_mask], calibrated_confs[~correct_mask],
              alpha=0.5, color='#C44E52', label=f'Incorrect (n={(~correct_mask).sum()})', s=30)
    
    # Diagonal line — points on this line had no change in confidence
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='No change line')
    ax.set_xlabel('Original Confidence', fontsize=12)
    ax.set_ylabel('Calibrated Confidence', fontsize=12)
    ax.set_title('Confidence Score Shift After Platt Scaling', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Platt Scaling Calibration')
    parser.add_argument('--file', type=str, default='outputs/evaluation_no_context.json',
                        help='Path to evaluation results JSON')
    parser.add_argument('--bins', type=int, default=10,
                        help='Number of bins for ECE calculation')
    args = parser.parse_args()

    predictions, confidences, true_labels = load_results(args.file)

    # ── Before calibration ──
    # Measure how well-calibrated the raw confidence scores are
    confs_raw = np.array(confidences) / 100.0
    correct = np.array([p == t for p, t in zip(predictions, true_labels)], dtype=float)

    ece_before, bins_before = compute_ece(correct, confs_raw, n_bins=args.bins)

    print("\n" + "=" * 70)
    print("PLATT SCALING CALIBRATION ANALYSIS")
    print("=" * 70)

    print(f"\n--- BEFORE CALIBRATION ---")
    print(f"Overall ECE: {ece_before:.4f}")
    print(f"Mean confidence: {confs_raw.mean():.3f}")
    print(f"Mean accuracy: {correct.mean():.3f}")
    print(f"Confidence range: {confs_raw.min():.2f} - {confs_raw.max():.2f}")

    # ── Apply Platt scaling (overall) ──
    # One logistic regression fitted on all predictions regardless of class
    print(f"\n--- APPLYING PLATT SCALING (overall) ---")
    calibrated_overall, correct_arr = apply_platt_scaling(predictions, confidences, true_labels)
    ece_after_overall, bins_after_overall = compute_ece(correct_arr.astype(float), calibrated_overall, n_bins=args.bins)

    print(f"ECE after overall Platt scaling: {ece_after_overall:.4f}")
    print(f"ECE improvement: {ece_before - ece_after_overall:.4f} ({(ece_before - ece_after_overall)/ece_before*100:.1f}%)")
    print(f"Calibrated confidence range: {calibrated_overall.min():.3f} - {calibrated_overall.max():.3f}")
    print(f"Calibrated mean confidence: {calibrated_overall.mean():.3f}")

    # ── Apply Platt scaling (per-class) ──
    # Separate logistic regression for each class (legitimate/spam/smishing)
    print(f"\n--- APPLYING PLATT SCALING (per-class) ---")
    calibrated_perclass, _ = apply_per_class_platt_scaling(predictions, confidences, true_labels)
    ece_after_perclass, bins_after_perclass = compute_ece(correct_arr.astype(float), calibrated_perclass, n_bins=args.bins)

    print(f"ECE after per-class Platt scaling: {ece_after_perclass:.4f}")
    print(f"ECE improvement: {ece_before - ece_after_perclass:.4f} ({(ece_before - ece_after_perclass)/ece_before*100:.1f}%)")
    print(f"Calibrated confidence range: {calibrated_perclass.min():.3f} - {calibrated_perclass.max():.3f}")
    print(f"Calibrated mean confidence: {calibrated_perclass.mean():.3f}")

    # ── Per-class ECE comparison ──
    # Show how each class's calibration changed under both methods
    print(f"\n--- PER-CLASS ECE COMPARISON ---")
    print(f"{'Class':<15} {'Before':<12} {'After (overall)':<18} {'After (per-class)':<18}")
    print("-" * 60)
    
    classes = sorted(set(predictions))
    for cls in classes:
        cls_mask = np.array([p == cls for p in predictions])
        cls_correct = correct[cls_mask]
        
        if len(cls_correct) == 0:
            continue
            
        cls_raw = confs_raw[cls_mask]
        cls_cal_overall = calibrated_overall[cls_mask]
        cls_cal_perclass = calibrated_perclass[cls_mask]
        
        ece_cls_before, _ = compute_ece(cls_correct, cls_raw, n_bins=args.bins)
        ece_cls_overall, _ = compute_ece(cls_correct.astype(float), cls_cal_overall, n_bins=args.bins)
        ece_cls_perclass, _ = compute_ece(cls_correct.astype(float), cls_cal_perclass, n_bins=args.bins)
        
        print(f"{cls:<15} {ece_cls_before:<12.4f} {ece_cls_overall:<18.4f} {ece_cls_perclass:<18.4f}")

    # ── Summary ──
    # Pick whichever method (overall vs per-class) gave lower ECE
    best_method = "overall" if ece_after_overall < ece_after_perclass else "per-class"
    best_ece = min(ece_after_overall, ece_after_perclass)
    best_bins = bins_after_overall if best_method == "overall" else bins_after_perclass
    best_calibrated = calibrated_overall if best_method == "overall" else calibrated_perclass
    
    print(f"\n--- SUMMARY ---")
    print(f"Best method: {best_method} Platt scaling")
    print(f"ECE: {ece_before:.4f} -> {best_ece:.4f} (improvement: {ece_before - best_ece:.4f})")
    
    if ece_before > 0:
        improvement_pct = (ece_before - best_ece) / ece_before * 100
        print(f"Relative improvement: {improvement_pct:.1f}%")
    
    # Interpret the final ECE using standard thresholds
    if best_ece < 0.05:
        print("Calibration quality: WELL CALIBRATED (ECE < 0.05)")
    elif best_ece < 0.10:
        print("Calibration quality: MODERATELY CALIBRATED (0.05 <= ECE < 0.10)")
    elif best_ece < 0.20:
        print("Calibration quality: POORLY CALIBRATED (0.10 <= ECE < 0.20)")
    else:
        print("Calibration quality: VERY POORLY CALIBRATED (ECE >= 0.20)")

    # ── Generate plots ──
    os.makedirs('outputs', exist_ok=True)

    # Side-by-side reliability diagrams (before vs after)
    plot_before_after(bins_before, ece_before, best_bins, best_ece,
                      title=f"Platt Scaling Calibration ({best_method.title()})",
                      save_path='outputs/platt_scaling_comparison.png')

    # Scatter plot showing how each prediction's confidence shifted
    plot_confidence_shift(confs_raw, best_calibrated, correct_arr,
                         save_path='outputs/platt_confidence_shift.png')

    print(f"\nGenerated plots:")
    print(f"  outputs/platt_scaling_comparison.png")
    print(f"  outputs/platt_confidence_shift.png")
    print("=" * 70)


if __name__ == "__main__":
    main()