# Hindi-English Code-Switching Model

This repository contains a fine-tuned XLM-RoBERTa model for Hindi-English code-switching. The model predicts masked tokens in mixed-language sentences, capturing natural patterns of language mixing across different demographics.

## Getting Started

**To quickly see the model in action, run the `RunMe.ipynb` Jupyter notebook.**

This will demonstrate the model's capabilities with code-switched sentences and provide demographic analysis of the predictions.

If you prefer to run the code yourself, see the examples below.

## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("lord-rajkumar/Code-Switch-Model")

# Create fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test with examples
test_example = "<mask>, kya scene hai?"  # Translation: <mask>, what's up?
results = fill_mask(test_example)

for result in results:
    print(f"Token: {result['token_str']}, Score: {result['score']:.4f}")
```

## With Demographic Analysis (Zero-Shot Classification)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("lord-rajkumar/Code-Switch-Model")

# Create a fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Create zero-shot classification pipeline for demographic analysis
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_demographics(token_str):
    """Classify the demographics of a token"""
    token_str_clean = token_str.strip()
    if not token_str_clean:
        return {"age": "unknown", "region": "unknown"}
    
    # Classify for age
    result_age = classifier(token_str_clean, candidate_labels=["under 30", "over 30"])
    age_label = result_age["labels"][0]
    age_score = result_age["scores"][0]
    
    # Classify for region
    result_region = classifier(token_str_clean, candidate_labels=["urban", "rural"])
    region_label = result_region["labels"][0]
    region_score = result_region["scores"][0]
    
    return {
        "age": age_label,
        "age_confidence": f"{age_score:.2f}",
        "region": region_label,
        "region_confidence": f"{region_score:.2f}"
    }

# Test with example sentences
examples = [
    "<mask>, kya scene hai?",   # Translation: <mask>, what's the scenario?
    "Project pe <mask> progress chal raha hai.", # Translation: <mask> the progress on the project?
    "Hello, <mask> kya kr raha hai?"    # Translation: Hello, <mask> what are you doing?
]

# Process each example
for example in examples:
    print(f"\n=== Input: {example} ===")
    results = fill_mask(example)
    for result in results:
        token = result['token_str']
        score = result['score']
        print(f"\nToken: '{token}', Score: {score:.4f}")
        
        # Perform demographic classification
        demographics = classify_demographics(token)
        print(f"  Demographics: Age likely {demographics['age']} (confidence: {demographics['age_confidence']})")
        print(f"               Region likely {demographics['region']} (confidence: {demographics['region_confidence']})")
```

## Model Information

- **Base Model**: XLM-RoBERTa
- **Task**: Masked Language Modeling (MLM)
- **Languages**: Hindi and English
- **HuggingFace Model**: [lord-rajkumar/Code-Switch-Model](https://huggingface.co/lord-rajkumar/Code-Switch-Model)

## Dataset

The model was trained on code-switched conversations across different demographics:
- Age groups: Teen, Adult, Senior
- Genders: Male, Female
- Regions: North, South, West of India

## Jupyter Notebook Demo

The easiest way to test the model is with the included Jupyter notebook:

1. Make sure you have Jupyter installed: `pip install jupyter`
2. Run the notebook: `jupyter notebook RunMe.ipynb`
3. Run the cells sequentially to see the model in action

The notebook demonstrates:
- Basic token prediction
- Demographic analysis using zero-shot classification
- Multiple examples with expected outputs

## Example Results with Demographic Analysis

### Example 1: Casual greeting
```
=== Input: <mask>, kya scene hai? ===

Token: 'Bhai', Score: 0.1594
Demographics: Age likely under 30 (confidence: 0.66)
              Region likely rural (confidence: 0.57)

Token: 'Hello', Score: 0.1397
Demographics: Age likely under 30 (confidence: 0.75)
              Region likely urban (confidence: 0.56)

Token: 'Hi', Score: 0.1270
Demographics: Age likely under 30 (confidence: 0.67)
              Region likely urban (confidence: 0.59)

Token: 'Sir', Score: 0.0762
Demographics: Age likely over 30 (confidence: 0.60)
              Region likely urban (confidence: 0.55)

Token: 'Hai', Score: 0.0436
Demographics: Age likely under 30 (confidence: 0.69)
              Region likely urban (confidence: 0.57)
```

### Example 2: Work context
```
=== Input: Project pe <mask> progress chal raha hai. ===

Token: 'kya', Score: 0.2187
Demographics: Age likely under 30 (confidence: 0.72)
              Region likely urban (confidence: 0.62)

Token: 'bahut', Score: 0.1086
Demographics: Age likely under 30 (confidence: 0.76)
              Region likely urban (confidence: 0.54)

Token: 'ek', Score: 0.0437
Demographics: Age likely under 30 (confidence: 0.68)
              Region likely urban (confidence: 0.53)

Token: 'mera', Score: 0.0393
Demographics: Age likely under 30 (confidence: 0.62)
              Region likely urban (confidence: 0.53)

Token: 'bhi', Score: 0.0346
Demographics: Age likely under 30 (confidence: 0.66)
              Region likely urban (confidence: 0.52)
```

### Example 3: Mixed language inquiry
```
=== Input: Hello, <mask> kya kr raha hai? ===

Token: 'aap', Score: 0.5082
Demographics: Age likely under 30 (confidence: 0.72)
              Region likely urban (confidence: 0.62)

Token: 'Aap', Score: 0.0889
Demographics: Age likely under 30 (confidence: 0.77)
              Region likely rural (confidence: 0.56)

Token: 'Abhi', Score: 0.0504
Demographics: Age likely under 30 (confidence: 0.75)
              Region likely urban (confidence: 0.50)

Token: 'Bhai', Score: 0.0452
Demographics: Age likely under 30 (confidence: 0.66)
              Region likely rural (confidence: 0.57)

Token: 'Rahul', Score: 0.0217
Demographics: Age likely under 30 (confidence: 0.72)
              Region likely urban (confidence: 0.85)
```

## Demographic Analysis Findings

The zero-shot classification reveals interesting patterns in the model's predictions:

1. **Age patterns**:
   - Most predicted tokens are classified as "under 30", which aligns with the prevalence of code-switching among younger generations
   - Formal terms like "Sir" are classified as "over 30", suggesting formality correlates with older age groups

2. **Regional patterns**:
   - English greetings like "Hello" and "Hi" are classified as more urban
   - Terms like "Bhai" have a higher rural classification than English equivalents
   - Personal names like "Rahul" have a very high urban confidence (0.85)

3. **Confidence levels**:
   - The model's confidence in age classification is generally higher than in regional classification
   - Most classifications have moderate confidence (0.55-0.75), which is appropriate for this type of analysis

These patterns suggest that code-switching has demographic dimensions that can be captured and analyzed using NLP techniques.

## Files Included

- `data-4.json` - The dataset used for training (for reference)
- `RunMe.ipynb` - Jupyter notebook for testing the model with demographic analysis

## Troubleshooting

If you encounter any issues:

1. **Model loading errors**: Ensure you have internet access to download from HuggingFace
2. **Missing packages**: Make sure transformers and torch are installed
3. **Version issues**: Try `pip install transformers==4.36.2` for compatibility

No training or complex setup is required to test this model. Simply run the code provided in the README or use the Jupyter notebook for interactive exploration.

## Authors ✨

- **[rapatel8](https://github.com/rapatel8)**
- **[KhatoonSaima](https://github.com/KhatoonSaima)**
- **[Chandravallika](https://github.com/Chandravallika)**