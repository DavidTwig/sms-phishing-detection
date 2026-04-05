"""
Adversarial Robustness Testing for SMS Phishing Detection
Tests detector resilience against four common text evasion techniques:
1. Homoglyph substitution (visually identical Unicode characters)
2. Leetspeak (letter-to-number/symbol substitution)
3. Character insertion (zero-width and invisible characters)
4. Synonym substitution (replacing key phishing words with synonyms)

Based on evasion techniques identified in:
- Unit 42 / Palo Alto Networks (2025), "The Homograph Illusion"
- Kavya & Sumathi (2024), "Staying ahead of phishers", Artificial Intelligence Review
- Wang et al. (2022), "Self-Consistency Improves Chain of Thought Reasoning"

Usage:
    python src/adversarial.py                    # Test all attacks on smishing messages
    python src/adversarial.py --attack homoglyph # Test specific attack
    python src/adversarial.py --n 50             # Test on 50 messages
    python src/adversarial.py --dataset corrected # Use corrected dataset
"""

import pandas as pd
import json
import os
import sys
import random
import argparse
from collections import Counter
from detector import SMSPhishingDetector

# Lock the random seed so results are the same every time the script runs
random.seed(42)

# ============================================================
# ATTACK 1: HOMOGLYPH SUBSTITUTION
# ============================================================
# Homoglyphs are characters from other alphabets (e.g. Cyrillic) that
# look identical to Latin letters but have different character codes.
# Used by attackers to bypass text-matching filters.
HOMOGLYPH_MAP = {
    'a': '\u0430',  # Cyrillic а
    'c': '\u0441',  # Cyrillic с
    'e': '\u0435',  # Cyrillic е
    'o': '\u043E',  # Cyrillic о
    'p': '\u0440',  # Cyrillic р
    'x': '\u0445',  # Cyrillic х
    'y': '\u0443',  # Cyrillic у
    'i': '\u0456',  # Cyrillic і
    's': '\u0455',  # Cyrillic ѕ
    'A': '\u0410',  # Cyrillic А
    'B': '\u0412',  # Cyrillic В
    'C': '\u0421',  # Cyrillic С
    'E': '\u0415',  # Cyrillic Е
    'H': '\u041D',  # Cyrillic Н
    'K': '\u041A',  # Cyrillic К
    'M': '\u041C',  # Cyrillic М
    'O': '\u041E',  # Cyrillic О
    'P': '\u0420',  # Cyrillic Р
    'T': '\u0422',  # Cyrillic Т
    'X': '\u0425',  # Cyrillic Х
}

def attack_homoglyph(text, intensity=0.3):
    """
    Replace a proportion of eligible characters with visually identical homoglyphs.
    
    Args:
        text: Original SMS text
        intensity: Proportion of eligible characters to replace (0.0 to 1.0)
    
    Returns:
        Modified text with homoglyph substitutions
    """
    chars = list(text)

    # Find every character position that has a homoglyph replacement available
    eligible_indices = [i for i, c in enumerate(chars) if c in HOMOGLYPH_MAP]
    
    # If no characters can be swapped, return the text unchanged
    if not eligible_indices:
        return text
    
    # Work out how many characters to replace based on the intensity percentage
    # Always replace at least 1 character so the attack actually does something
    n_replace = max(1, int(len(eligible_indices) * intensity))
    replace_indices = random.sample(eligible_indices, min(n_replace, len(eligible_indices)))
    
    # Swap each selected character with its Cyrillic/Greek lookalike
    for i in replace_indices:
        chars[i] = HOMOGLYPH_MAP[chars[i]]
    
    return ''.join(chars)


# ============================================================
# ATTACK 2: LEETSPEAK SUBSTITUTION
# ============================================================
# Leetspeak replaces letters with similar-looking numbers/symbols
# e.g. "password" becomes "p4$$w0rd". A technique used in phishing
# messages to dodge keyword-based filters.
LEET_MAP = {
    'a': '4', 'A': '4',
    'e': '3', 'E': '3',
    'i': '1', 'I': '1',
    'o': '0', 'O': '0',
    's': '$', 'S': '$',
    't': '7', 'T': '7',
    'l': '1', 'L': '1',
    'b': '8', 'B': '8',
    'g': '9', 'G': '9',
}

def attack_leetspeak(text, intensity=0.3):
    """
    Replace a proportion of eligible characters with leetspeak equivalents.
    
    Args:
        text: Original SMS text
        intensity: Proportion of eligible characters to replace
    
    Returns:
        Modified text with leetspeak substitutions
    """
    chars = list(text)
    eligible_indices = [i for i, c in enumerate(chars) if c in LEET_MAP]
    
    if not eligible_indices:
        return text
    
    # Same logic as homoglyph — pick a random subset of eligible characters to replace
    n_replace = max(1, int(len(eligible_indices) * intensity))
    replace_indices = random.sample(eligible_indices, min(n_replace, len(eligible_indices)))
    
    for i in replace_indices:
        chars[i] = LEET_MAP[chars[i]]
    
    return ''.join(chars)


# ============================================================
# ATTACK 3: CHARACTER INSERTION
# ============================================================
# Unicode characters that take up zero visual space. Inserting them
# into words makes the text look normal but the underlying data is
# different, which can confuse text-matching detectors.
INVISIBLE_CHARS = [
    '\u200B',  # Zero-width space
    '\u200C',  # Zero-width non-joiner
    '\u200D',  # Zero-width joiner
    '\uFEFF',  # Zero-width no-break space
    '\u00AD',  # Soft hyphen
]

def attack_char_insertion(text, intensity=0.15):
    """
    Insert invisible/zero-width Unicode characters between letters of key words.
    
    Args:
        text: Original SMS text
        intensity: Proportion of character positions to insert at
    
    Returns:
        Modified text with invisible characters inserted
    """
    chars = list(text)
    # Only pick positions between two letters — inserting next to
    # spaces or punctuation wouldn't break up any words
    eligible_indices = [i for i in range(1, len(chars)) 
                        if chars[i-1].isalpha() and chars[i].isalpha()]
    
    if not eligible_indices:
        return text
    
    n_insert = max(1, int(len(eligible_indices) * intensity))
    # Sort in reverse order so that inserting characters doesn't shift
    # the positions of characters that haven't been processed yet
    insert_indices = sorted(random.sample(eligible_indices, min(n_insert, len(eligible_indices))), 
                           reverse=True)  # Reverse to maintain index validity
    
    for i in insert_indices:
        invisible_char = random.choice(INVISIBLE_CHARS)
        chars.insert(i, invisible_char)
    
    return ''.join(chars)


# ============================================================
# ATTACK 4: SYNONYM SUBSTITUTION
# ============================================================
# Replaces common phishing trigger words with synonyms (e.g. "verify"
# becomes "confirm"). If a detector relies on specific keywords, this
# could cause it to miss the phishing attempt. The most "natural"
# attack since the message still reads normally.
SYNONYM_MAP = {
    'verify': ['confirm', 'validate', 'authenticate', 'check'],
    'account': ['profile', 'membership', 'registration'],
    'suspended': ['restricted', 'limited', 'paused', 'frozen', 'disabled'],
    'urgent': ['immediate', 'critical', 'important', 'pressing'],
    'security': ['protection', 'safety', 'defense'],
    'password': ['passcode', 'PIN', 'access code', 'credentials'],
    'bank': ['financial institution', 'banking service'],
    'update': ['refresh', 'renew', 'modify', 'change'],
    'click': ['tap', 'visit', 'go to', 'open', 'follow'],
    'confirm': ['verify', 'validate', 'acknowledge'],
    'login': ['sign in', 'log on', 'access'],
    'expire': ['run out', 'lapse', 'end', 'terminate'],
    'delivery': ['shipment', 'package', 'parcel', 'dispatch'],
    'payment': ['transaction', 'transfer', 'charge'],
    'blocked': ['restricted', 'locked', 'frozen', 'held'],
    'unusual': ['suspicious', 'unexpected', 'irregular', 'abnormal'],
    'immediately': ['right away', 'at once', 'now', 'without delay'],
    'warning': ['notice', 'alert', 'notification', 'advisory'],
    'refund': ['reimbursement', 'return', 'credit'],
    'unauthorized': ['unapproved', 'illegitimate', 'unverified'],
}

def attack_synonym(text, intensity=0.5):
    """
    Replace key phishing-related words with synonyms.
    
    Args:
        text: Original SMS text
        intensity: Proportion of eligible words to replace
    
    Returns:
        Modified text with synonym substitutions
    """
    words = text.split()
    eligible_indices = []
    
    # Go through each word, strip punctuation, and check if it's
    # one of the phishing keywords that has synonym replacements
    for i, word in enumerate(words):
        # Strip punctuation for matching
        clean_word = word.lower().strip('.,!?:;()[]{}"\'-')
        if clean_word in SYNONYM_MAP:
            eligible_indices.append((i, clean_word))
    
    if not eligible_indices:
        return text
    
    # Pick a random subset of the matched words to replace
    n_replace = max(1, int(len(eligible_indices) * intensity))
    replace_items = random.sample(eligible_indices, min(n_replace, len(eligible_indices)))
    
    for idx, clean_word in replace_items:
        original_word = words[idx]
        synonym = random.choice(SYNONYM_MAP[clean_word])
        
        # Preserve capitalisation pattern
        # e.g. if original was "Verify", synonym becomes "Confirm" (not "confirm")
        if original_word[0].isupper():
            synonym = synonym.capitalize()
        if original_word.isupper():
            synonym = synonym.upper()
        
        # Preserve trailing punctuation
        # e.g. if original was "account!" the result should be "profile!" not just "profile"
        trailing = ''
        while original_word and not original_word[-1].isalpha():
            trailing = original_word[-1] + trailing
            original_word = original_word[:-1]
        
        words[idx] = synonym + trailing
    
    return ' '.join(words)


# ============================================================
# ATTACK REGISTRY
# ============================================================
# Maps each attack name to its function, description, and intensity
# levels. Intensities differ per attack — e.g. char_insertion uses
# lower values (0.1-0.3) since too many invisible characters is unrealistic.
ATTACKS = {
    'homoglyph': {
        'function': attack_homoglyph,
        'description': 'Replace characters with visually identical Unicode homoglyphs',
        'intensities': [0.1, 0.3, 0.5]
    },
    'leetspeak': {
        'function': attack_leetspeak,
        'description': 'Replace letters with numbers/symbols (e.g., a->4, e->3)',
        'intensities': [0.1, 0.3, 0.5]
    },
    'char_insertion': {
        'function': attack_char_insertion,
        'description': 'Insert invisible zero-width Unicode characters between letters',
        'intensities': [0.1, 0.2, 0.3]
    },
    'synonym': {
        'function': attack_synonym,
        'description': 'Replace key phishing words with synonyms',
        'intensities': [0.3, 0.5, 1.0]
    }
}


# ============================================================
# EVALUATION
# ============================================================
def evaluate_attack(detector, messages, labels, attack_name, attack_func, intensity):
    """
    Apply an attack to messages and evaluate detector performance.
    
    Returns:
        Dictionary with results
    """
    original_correct = 0
    attacked_correct = 0
    attack_evaded = 0  # Was correct before attack, wrong after
    predictions_original = []
    predictions_attacked = []
    
    for sms, true_label in zip(messages, labels):
        # Step 1: Classify the original (unmodified) message to get a baseline
        try:
            orig_result = detector.detect(sms)
            orig_pred = orig_result['classification']
        except Exception as e:
            orig_pred = 'unknown'
        
        # Step 2: Apply the attack to modify the message text
        attacked_sms = attack_func(sms, intensity=intensity)
        
        # Step 3: Classify the attacked (modified) message
        try:
            attack_result = detector.detect(attacked_sms)
            attack_pred = attack_result['classification']
        except Exception as e:
            attack_pred = 'unknown'
        
        predictions_original.append(orig_pred)
        predictions_attacked.append(attack_pred)
        
        # Count up correct predictions for both original and attacked versions
        if orig_pred == true_label:
            original_correct += 1
        if attack_pred == true_label:
            attacked_correct += 1
        # Track cases where the attack fooled the detector
        # (got the original right but the attacked version wrong)
        if orig_pred == true_label and attack_pred != true_label:
            attack_evaded += 1
    
    n = len(messages)
    original_acc = original_correct / n if n > 0 else 0
    attacked_acc = attacked_correct / n if n > 0 else 0
    # Evasion rate = what percentage of correctly-detected messages were
    # flipped to incorrect by the attack (the key metric for robustness)
    evasion_rate = attack_evaded / original_correct if original_correct > 0 else 0
    
    return {
        'attack': attack_name,
        'intensity': intensity,
        'n_messages': n,
        'original_accuracy': original_acc,
        'attacked_accuracy': attacked_acc,
        'accuracy_drop': original_acc - attacked_acc,
        'evasion_rate': evasion_rate,
        'evaded_count': attack_evaded,
        'original_correct': original_correct,
        'predictions_original': predictions_original,
        'predictions_attacked': predictions_attacked
    }


def print_attack_examples(messages, attack_func, intensity, n_examples=3):
    """Show examples of attack transformations."""
    print(f"\n  Example transformations (intensity={intensity}):")
    for i, sms in enumerate(messages[:n_examples]):
        # Truncate long messages to 80 characters for readable output
        original = sms[:80] + ('...' if len(sms) > 80 else '')
        attacked = attack_func(sms, intensity=intensity)
        attacked_preview = attacked[:80] + ('...' if len(attacked) > 80 else '')
        print(f"    [{i+1}] Original: {original}")
        print(f"        Attacked: {attacked_preview}")
    print()


# ============================================================
# MAIN
# ============================================================
def main():
    # Set up command line arguments to control what gets tested
    # without editing the code each time
    parser = argparse.ArgumentParser(description='Adversarial Robustness Testing')
    parser.add_argument('--attack', type=str, default='all',
                        choices=['all', 'homoglyph', 'leetspeak', 'char_insertion', 'synonym'],
                        help='Which attack to test (default: all)')
    parser.add_argument('--n', type=int, default=None,
                        help='Number of smishing messages to test (default: all)')
    parser.add_argument('--dataset', type=str, default='corrected',
                        choices=['original', 'corrected'],
                        help='Which dataset to use (default: corrected)')
    args = parser.parse_args()
    
    # Load the chosen dataset — defaults to corrected (the one with
    # 97 mislabelled smishing messages fixed to spam)
    if args.dataset == 'corrected':
        dataset_path = 'data/dataset_corrected.csv'
    else:
        dataset_path = 'data/dataset.csv'
    
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at '{dataset_path}'")
        sys.exit(1)
    
    df = pd.read_csv(dataset_path)
    # Normalise labels to lowercase and strip whitespace so matching works
    df['label'] = df['label'].str.strip().str.lower()
    
    # Only test on smishing messages — the point is to see if attacks
    # can make the detector miss actual phishing attempts.
    # Spam/legitimate aren't the safety-critical class.
    smishing_df = df[df['label'] == 'smishing'].copy()
    
    # Optionally take a random sample if --n was specified
    # (useful for quick tests without burning through the Groq API quota)
    if args.n:
        smishing_df = smishing_df.sample(n=min(args.n, len(smishing_df)), random_state=42)
    
    messages = smishing_df['SMS'].tolist()
    labels = smishing_df['label'].tolist()
    
    print("=" * 70)
    print("ADVERSARIAL ROBUSTNESS TESTING")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Smishing messages to test: {len(messages)}")
    print(f"Attack(s): {args.attack}")
    
    # Set up the detector with the same config as the main evaluation:
    # no context gathering (it was found to hurt accuracy) and two-pass enabled
    print("\nInitialising detector...")
    detector = SMSPhishingDetector(use_context=False, use_two_pass=True)
    
    # Determine which attacks to run
    if args.attack == 'all':
        attacks_to_run = list(ATTACKS.keys())
    else:
        attacks_to_run = [args.attack]
    
    # Store all results to print a summary table at the end
    all_results = []
    
    for attack_name in attacks_to_run:
        attack_info = ATTACKS[attack_name]
        attack_func = attack_info['function']
        
        print("\n" + "=" * 70)
        print(f"ATTACK: {attack_name.upper()}")
        print(f"Description: {attack_info['description']}")
        print("=" * 70)
        
        # Show before/after examples of what the attack does to the text
        # (uses the middle intensity level)
        print_attack_examples(messages, attack_func, attack_info['intensities'][1])
        
        # Test at each intensity level (e.g. 10%, 30%, 50%) to see how
        # the detector holds up as attacks get more aggressive
        for intensity in attack_info['intensities']:
            print(f"  Testing intensity={intensity}...")
            result = evaluate_attack(detector, messages, labels, 
                                    attack_name, attack_func, intensity)
            all_results.append(result)
            
            print(f"    Original accuracy: {result['original_accuracy']:.1%}")
            print(f"    Attacked accuracy: {result['attacked_accuracy']:.1%}")
            print(f"    Accuracy drop:     {result['accuracy_drop']:.1%}")
            print(f"    Evasion rate:      {result['evasion_rate']:.1%} "
                  f"({result['evaded_count']}/{result['original_correct']} correctly detected messages evaded)")
    
    # Print a final summary table with all attacks and intensities side by side
    print("\n" + "=" * 70)
    print("SUMMARY: ADVERSARIAL ROBUSTNESS RESULTS")
    print("=" * 70)
    print(f"{'Attack':<18} {'Intensity':<12} {'Original':<12} {'Attacked':<12} {'Drop':<10} {'Evasion Rate':<15}")
    print("-" * 70)
    
    for r in all_results:
        print(f"{r['attack']:<18} {r['intensity']:<12} {r['original_accuracy']:<12.1%} "
              f"{r['attacked_accuracy']:<12.1%} {r['accuracy_drop']:<10.1%} {r['evasion_rate']:<15.1%}")
    
    # Save results to JSON for dissertation tables/figures
    os.makedirs('outputs', exist_ok=True)
    # Only save numeric results, not full prediction lists
    save_data = []
    for r in all_results:
        save_data.append({
            'attack': r['attack'],
            'intensity': r['intensity'],
            'n_messages': r['n_messages'],
            'original_accuracy': r['original_accuracy'],
            'attacked_accuracy': r['attacked_accuracy'],
            'accuracy_drop': r['accuracy_drop'],
            'evasion_rate': r['evasion_rate'],
            'evaded_count': r['evaded_count']
        })
    
    with open('outputs/adversarial_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to outputs/adversarial_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()