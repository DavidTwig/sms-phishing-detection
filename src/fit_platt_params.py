"""
Fit Platt Scaling Parameters
Fits a logistic regression on the full evaluation data and saves
the parameters for integration into detector.py.

Usage: python src/fit_platt_params.py --file outputs/evaluation_no_context.json

This only needs to be run once. The output parameters are then
hardcoded into detector.py.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description='Fit Platt Scaling Parameters')
    parser.add_argument('--file', type=str, default='outputs/evaluation_no_context.json',
                        help='Path to evaluation results JSON')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}")
        sys.exit(1)

    with open(args.file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    predictions = data['predictions']
    confidences = data['confidences']
    true_labels = data['true_labels']

    # Normalise confidence from 0-100 to 0-1 for the logistic regression
    confs = np.array(confidences) / 100.0
    # Create a 1/0 array: 1 = prediction was correct, 0 = incorrect
    correct = np.array([p == t for p, t in zip(predictions, true_labels)], dtype=int)

    print(f"Loaded {len(predictions)} predictions")
    print(f"Accuracy: {correct.mean():.3f}")
    print(f"Raw confidence range: {confs.min():.2f} - {confs.max():.2f}")
    print(f"Raw mean confidence: {confs.mean():.3f}")

    # Fit a logistic regression that learns the relationship between
    # raw confidence scores and whether the prediction was actually correct.
    # This produces two numbers (coefficient and intercept) that define
    # a sigmoid curve mapping raw confidence to calibrated confidence.
    # reshape(-1, 1) converts the 1D array into a 2D column because
    # scikit-learn expects input shaped as (n_samples, n_features).
    lr = LogisticRegression(random_state=42)
    lr.fit(confs.reshape(-1, 1), correct)

    # The coefficient and intercept are the two Platt scaling parameters.
    # These get hardcoded into detector.py as PLATT_COEF and PLATT_INTERCEPT.
    coef = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    print(f"\nPlatt scaling parameters:")
    print(f"  Coefficient (A): {coef:.6f}")
    print(f"  Intercept (B):   {intercept:.6f}")
    # The calibration formula is a sigmoid function:
    # Take the raw score, multiply by A, add B, then pass through
    # 1/(1+e^-x) which squashes the result into the 0-1 range.
    # This adjusts overconfident scores downward.
    print(f"\n  Formula: calibrated = sigmoid(A * raw_confidence + B)")
    print(f"  calibrated = 1 / (1 + exp(-(A * conf/100 + B)))")

    # Sanity check — apply the calibration to see the adjusted range
    calibrated = 1 / (1 + np.exp(-(coef * confs + intercept)))
    print(f"\n  Calibrated range: {calibrated.min():.3f} - {calibrated.max():.3f}")
    print(f"  Calibrated mean:  {calibrated.mean():.3f}")

    # Show what happens to specific confidence values after calibration
    # e.g. a raw 95% might become 88% after Platt scaling
    print(f"\n  Example mappings (raw -> calibrated):")
    for raw in [80, 85, 90, 95, 100]:
        cal = 1 / (1 + np.exp(-(coef * (raw/100) + intercept)))
        print(f"    {raw}% -> {cal*100:.1f}%")

    print(f"\nCopy these values into detector.py:")
    print(f"  PLATT_COEF = {coef:.6f}")
    print(f"  PLATT_INTERCEPT = {intercept:.6f}")


if __name__ == "__main__":
    main()