# Hindi-English Code-Switching Model

A fine-tuned XLM-RoBERTa model for Hindi-English code-switching prediction and demographic analysis.

> **Quick Start :** To instantly test this model, simply run the `RunMe.ipynb` Jupyter notebook included in this repository. All the necessary code and examples are ready to execute.

## Project Overview

This project implements a Masked Language Model (MLM) that understands and predicts Hindi-English code-switching patterns across different demographic groups. The model can predict masked tokens in mixed-language sentences and analyze the demographic properties of these predictions.

![Model Workflow](/docs/workflow-diagram.svg)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hindi-english-code-switching.git
cd hindi-english-code-switching

# Install dependencies
pip install -r requirements.txt

# All set! Now, open and run RunMe.ipynb
```

## Architecture

The model architecture consists of:

1. **Base Model**: XLM-RoBERTa (multilingual transformer model)
   - 12 encoder layers
   - 768 hidden dimensions
   - 12 attention heads
   - 250K vocabulary tokens

2. **Driver Components**:
   - **Tokenizer**: XLM-RoBERTa tokenizer with SentencePiece
   - **MLM Head**: Linear layer for masked token prediction
   - **Zero-Shot Classifier**: BART-large-mnli for demographic analysis

3. **Data Flow**:
   - Input text → Tokenization → Token embeddings → Transformer layers → MLM prediction
   - Predicted tokens → Zero-shot classifier → Demographic properties

## FAIR-Compliant Code Structure

```
hindi-english-code-switching/
├── data/
│   └── data-4.json            # Training dataset
├── models/
│   └── README.md              # Points to HuggingFace model
├── notebooks/
│   └── RunMe.ipynb            # Demo notebook
├── src/
│   ├── train.py               # Training script
│   └── test.py                # Testing/evaluation script
├── docs/
│   └── workflow-diagram.svg   # Visualization of model workflow
├── README.md                  # Project documentation
├── requirements.txt           # Dependencies
└── LICENSE                    # MIT License
```

### Code Features

- **Modular Design**: Separates training, testing, and evaluation components
- **Reproducibility**: Fixed random seeds and documented hyperparameters
- **Interoperability**: Compatible with the HuggingFace ecosystem
- **Documentation**: Inline comments and function docstrings

## Quickstart

Run the provided Jupyter notebook to instantly see the model in action:

```bash
jupyter notebook notebooks/RunMe.ipynb
```

Or use the model with just a few lines of code:

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained("lord-rajkumar/Code-Switch-Model")

# Create fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Test with an example
results = fill_mask("<mask>, kya scene hai?")  # "<mask>, what's up?"
for result in results:
    print(f"Token: {result['token_str']}, Score: {result['score']:.4f}")
```

## Reproducible Results

The model consistently produces the following results for our test examples:

### Example 1: `<mask>, kya scene hai?`
```
Token: 'Bhai', Score: 0.1594
Token: 'Hello', Score: 0.1397
Token: 'Hi', Score: 0.1270
Token: 'Sir', Score: 0.0762
Token: 'Hai', Score: 0.0436
```

### Example 2: `Project pe <mask> progress chal raha hai.`
```
Token: 'kya', Score: 0.2187
Token: 'bahut', Score: 0.1086
Token: 'ek', Score: 0.0437
Token: 'mera', Score: 0.0393
Token: 'bhi', Score: 0.0346
```

### Example 3: `Hello, <mask> kya kr raha hai?`
```
Token: 'aap', Score: 0.5082
Token: 'Aap', Score: 0.0889
Token: 'Abhi', Score: 0.0504
Token: 'Bhai', Score: 0.0452
Token: 'Rahul', Score: 0.0217
```

## Demographic Analysis

The zero-shot classification reveals interesting patterns in the model's predictions:

1. **Age patterns**:
   - Most predicted tokens are classified as "under 30"
   - Formal terms like "Sir" are classified as "over 30"

2. **Regional patterns**:
   - English greetings like "Hello" and "Hi" are classified as more urban
   - Terms like "Bhai" have a higher rural classification

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Acknowledgments

- HuggingFace for the transformers library
- XLM-RoBERTa authors for the base model

## Authors ✨

- **[Rajkumar](https://github.com/rajpatel8)**
- **[KhatoonSaima](https://github.com/KhatoonSaima)**
- **[Chandravallika](https://github.com/Chandravallika)**