# SMS Phishing Detection with Confidence-Aware Explainability

An SMS phishing detection system using Llama 3.3 70B Versatile via Groq's free API tier. Classifies SMS messages as **legitimate**, **spam**, or **smishing** using a two-pass classification architecture with Platt-scaled confidence scores.

Achieves **93.8% accuracy** on a corrected version of the SmishX dataset (Wang et al., SOUPS 2025), within 5 percentage points of the proprietary GPT-4o-based SmishX system (98.8%).

**Author:** David Twigger  
**Module:** CM3203 — One Semester Individual Project   
**University:** Cardiff University  
**Supervisor:** Dr. Saxena

---

## Setup

### Prerequisites

- Python 3
- A Groq API key (free tier): https://console.groq.com/

### Installation

```bash
cd sms-phishing-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, use `source venv/bin/activate` instead of `venv\Scripts\activate`.

### Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

### Verify Setup

```bash
python src/test_setup.py
```

This checks that all packages are installed and the Groq API key is working.

---

## Usage

**All commands should be run from the project root (`sms-phishing-detection/`) with the virtual environment activated.**

### Quick Test

```bash
python src/test_detector.py                  # Test on 10 balanced messages with context
python src/test_detector.py -n 20            # Test on 20 messages
python src/test_detector.py --no-context     # Test without context gathering
python src/test_detector.py --unbalanced     # Random sample instead of balanced across classes
```

Show's the main functionality of the system. For each message, this prints the predicted classification, confidence percentage, and the LLM's natural language explanation of its reasoning. It also shows whether each prediction was correct or wrong, and prints an overall accuracy summary with a per-class breakdown at the end.

### Full Evaluation

```bash
python src/evaluate.py --no-context          # Full evaluation (1,200 messages, no context)
python src/evaluate.py --no-context --sample 100   # Quick test on 100 messages
python src/evaluate.py                       # Full evaluation with context gathering (slow)
```

Results are saved to `outputs/evaluation_no_context.json` or `outputs/evaluation_with_context.json`.

**Note:** The dataset path in `evaluate.py` needs to be manually switched between `data/dataset.csv` (original) and `data/dataset_corrected.csv` (corrected) depending on which version to evaluate.

### Adversarial Robustness Testing

```bash
python src/adversarial.py                          # All 4 attacks on all smishing messages
python src/adversarial.py --attack homoglyph       # Test only homoglyph attack
python src/adversarial.py --attack leetspeak       # Test only leetspeak attack
python src/adversarial.py --attack char_insertion  # Test only character insertion attack
python src/adversarial.py --attack synonym         # Test only synonym substitution attack
python src/adversarial.py --n 50                   # Test on 50 smishing messages (faster)
python src/adversarial.py --dataset original       # Use original dataset instead of corrected
```

Results are saved to `outputs/adversarial_results.json`.

### Confidence Calibration Analysis

```bash
python src/calibration.py                                          # Default (10 bins)
python src/calibration.py --file outputs/evaluation_no_context.json  # Specify results file
python src/calibration.py --bins 15                                 # Use 15 bins instead of 10
```

Generates:
- `outputs/reliability_diagram.png`
- `outputs/reliability_per_class.png`
- `outputs/confidence_distribution.png`

### Platt Scaling Analysis

```bash
python src/platt_scaling.py                                          # Default
python src/platt_scaling.py --file outputs/evaluation_no_context.json  # Specify results file
```

Generates:
- `outputs/platt_scaling_comparison.png`
- `outputs/platt_confidence_shift.png`

### Fit Platt Scaling Parameters

```bash
python src/fit_platt_params.py --file outputs/evaluation_no_context.json
```

Outputs the PLATT_COEF and PLATT_INTERCEPT values to copy into `detector.py`. Only needs to be run once.

### Error Analysis

```bash
python src/error_analysis.py
python src/error_analysis.py > outputs/error_analysis_report.txt   # Save output to file
```

**Note:** The `RESULTS_PATH` and `DATASET_PATH` variables at the top of `error_analysis.py` need to be set to match whichever evaluation run and dataset version is being analysed.

Generates `outputs/error_analysis.csv` for manual review in Excel.

### Dataset Correction

```bash
python src/create_corrected_dataset.py
```

Creates `data/dataset_corrected.csv` by relabelling 97 mislabelled smishing messages to spam. Requires `outputs/evaluation_no_context.json` to exist (run evaluation first).

### Dataset Exploration

```bash
python src/explore_data.py
```

Prints an overview of the SmishX dataset structure, columns, data types, and label distribution.

### Manual Audit Scripts

```bash
python src/extract_smishing_spam.py     # Extract smishing→spam misclassifications for audit
python src/verify_relabelled.py         # View all relabelled messages for verification
```

---

## Final System Configuration

- **Model:** Llama 3.3 70B Versatile via Groq API
- **Classification:** Two-pass system (first pass classifies, second pass reviews flagged messages)
- **Temperature:** 0.1
- **Context gathering:** Disabled (found to hurt accuracy)
- **Few-shot examples:** Disabled (found to degrade performance)
- **Confidence calibration:** Platt scaling (PLATT_COEF = 3.121423, PLATT_INTERCEPT = -0.178857)
- **Best accuracy:** 93.8% on corrected dataset