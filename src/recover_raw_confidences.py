"""
One-time utility to recover raw (pre-Platt) confidence scores from
an existing evaluation JSON where detector.py applied Platt scaling
in-line. Adds a 'confidences_raw' field without re-running evaluation.
"""
import json
import numpy as np
import os

INPUT = 'outputs/evaluation_no_context.json'
PLATT_COEF = 3.121423
PLATT_INTERCEPT = -0.178857

# Build the forward mapping: raw -> calibrated (as integer, matching
# how detector.py stores it after rounding)
mapping_to_raw = {}
for raw in [80, 85, 90, 95, 100]:
    cal_pct = 100 * (1 / (1 + np.exp(-(PLATT_COEF * (raw/100) + PLATT_INTERCEPT))))
    mapping_to_raw[round(cal_pct)] = raw
# Result: {91: 80, 92: 85, 93: 90, 94: 95, 95: 100}

print("Calibrated -> Raw mapping:", mapping_to_raw)

with open(INPUT, 'r', encoding='utf-8') as f:
    data = json.load(f)

confidences = data['confidences']

# Stats on what's in the file
from collections import Counter
print("\nDistribution of confidence values in JSON:")
for v, n in sorted(Counter(confidences).items()):
    note = "(Platt-scaled, raw=" + str(mapping_to_raw[v]) + ")" if v in mapping_to_raw \
           else "(raw second-pass override)"
    print(f"  {v}: {n:>4} predictions  {note}")

# Recover raw values
recovered = []
n_inverted = 0
n_passthrough = 0
for v in confidences:
    if v in mapping_to_raw:
        recovered.append(mapping_to_raw[v])
        n_inverted += 1
    else:
        # Already-raw second-pass override score, keep as is
        recovered.append(v)
        n_passthrough += 1

print(f"\nInverted Platt scaling on {n_inverted} predictions")
print(f"Passed through {n_passthrough} second-pass override predictions unchanged")

# Add the recovered field and write back
data['confidences_raw'] = recovered

with open(INPUT, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)

print(f"\nUpdated {INPUT} with confidences_raw field")
print(f"Mean raw: {np.mean(recovered):.2f}, Range: {min(recovered)} - {max(recovered)}")