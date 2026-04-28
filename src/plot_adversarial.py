"""
plot_adversarial.py

Generates a grouped bar chart of original vs attacked accuracy across all
four adversarial attack types and three intensity levels each.

Reads outputs/adversarial_results.json (produced by adversarial.py) and
saves outputs/adversarial_results_chart.png for inclusion in the dissertation
as Figure 5.5.

Usage:
    python src/plot_adversarial.py
"""
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = 'outputs/adversarial_results.json'
OUTPUT_PATH = 'outputs/adversarial_results_chart.png'

# Display order and human-readable labels for the four attacks
ATTACK_ORDER = ['homoglyph', 'leetspeak', 'char_insertion', 'synonym']
ATTACK_LABELS = {
    'homoglyph': 'Homoglyph',
    'leetspeak': 'Leetspeak',
    'char_insertion': 'Char insertion',
    'synonym': 'Synonym',
}


def load_results(path):
    """Load adversarial results JSON, sorted by attack and intensity."""
    if not os.path.exists(path):
        print(f"ERROR: Results file not found at '{path}'")
        print("Run adversarial.py first to generate the results.")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Group by attack type, preserving intensity order from the JSON
    grouped = {a: [] for a in ATTACK_ORDER}
    for entry in data:
        attack = entry['attack']
        if attack in grouped:
            grouped[attack].append(entry)

    # Sort each attack's results by intensity (ascending)
    for attack in grouped:
        grouped[attack].sort(key=lambda e: e['intensity'])

    return grouped


def plot_results(grouped, output_path):
    """Build the grouped bar chart and save it."""

    # Flatten into a single ordered list of (attack, intensity, orig, atk)
    bars = []
    group_starts = []  # x-position where each attack group starts
    idx = 0
    for attack in ATTACK_ORDER:
        if not grouped[attack]:
            continue
        group_starts.append((attack, idx))
        for entry in grouped[attack]:
            bars.append({
                'attack': attack,
                'intensity': entry['intensity'],
                'original': entry['original_accuracy'] * 100,
                'attacked': entry['attacked_accuracy'] * 100,
            })
            idx += 1

    n = len(bars)
    x = np.arange(n)
    bar_w = 0.38

    fig, ax = plt.subplots(figsize=(13, 6.5))

    # Plot original (light blue) and attacked (red) bars side by side
    color_orig = '#9DB4D6'
    color_atk = '#C44E52'
    edge = '#2E4057'

    for i, b in enumerate(bars):
        ax.bar(x[i] - bar_w / 2, b['original'], bar_w,
               color=color_orig, edgecolor=edge, linewidth=0.6,
               label='Original' if i == 0 else "")
        ax.bar(x[i] + bar_w / 2, b['attacked'], bar_w,
               color=color_atk, edgecolor=edge, linewidth=0.6,
               label='Attacked' if i == 0 else "")
        # Value labels above each bar
        ax.text(x[i] - bar_w / 2, b['original'] + 0.15, f"{b['original']:.1f}",
                ha='center', fontsize=8.5, color='#1a1a1a')
        ax.text(x[i] + bar_w / 2, b['attacked'] + 0.15, f"{b['attacked']:.1f}",
                ha='center', fontsize=8.5, color='#1a1a1a')

    # X-axis: intensity values as tick labels
    intensity_labels = [str(b['intensity']) for b in bars]
    ax.set_xticks(x)
    ax.set_xticklabels(intensity_labels, fontsize=10)
    ax.set_xlabel('Attack Intensity', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)

    # Y-axis: zoomed in to make differences visible (don't start at 0)
    all_vals = [b['original'] for b in bars] + [b['attacked'] for b in bars]
    y_min = max(85, min(all_vals) - 3)
    y_max = 105
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(int(y_min // 5) * 5, 101, 5))

    # Attack-name banners and separator lines between groups
    for i, (attack, start) in enumerate(group_starts):
        n_intensities = len(grouped[attack])
        centre = start + (n_intensities - 1) / 2
        ax.text(centre, y_max - 1.5, ATTACK_LABELS[attack],
                ha='center', fontsize=11, fontweight='bold',
                color='#1f3a5f')
        # Vertical separator after this group (except for the last one)
        if i < len(group_starts) - 1:
            ax.axvline(start + n_intensities - 0.5,
                       color='gray', linestyle=':', alpha=0.5)

    # Get total smishing message count from first entry (all should match)
    n_messages = next(iter(next(iter(grouped.values()), [{'n_messages': '?'}])),
                      {'n_messages': '?'})['n_messages']

    ax.set_title(
        f'Detection Accuracy Under Adversarial Attacks '
        f'(Smishing Class, n={n_messages})',
        fontsize=13, pad=12
    )
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary(grouped):
    """Print a summary of the values plotted, for sanity checking."""
    print("\n" + "=" * 60)
    print("VALUES PLOTTED")
    print("=" * 60)
    print(f"{'Attack':<16} {'Intensity':<10} {'Original':<12} {'Attacked':<12} {'Drop':<8}")
    print("-" * 60)
    for attack in ATTACK_ORDER:
        for entry in grouped[attack]:
            orig = entry['original_accuracy'] * 100
            atk = entry['attacked_accuracy'] * 100
            drop = orig - atk
            print(f"{ATTACK_LABELS[attack]:<16} {entry['intensity']:<10} "
                  f"{orig:<12.1f} {atk:<12.1f} {drop:<8.1f}")
    print("=" * 60)


def main():
    grouped = load_results(INPUT_PATH)
    print_summary(grouped)
    plot_results(grouped, OUTPUT_PATH)
    print(f"\nFigure ready for dissertation as Figure 5.5.")


if __name__ == "__main__":
    main()