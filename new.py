from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load your custom fine-tuned model and tokenizer
model_path = "./code_switch_mlm/checkpoint-50"
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained(model_path)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Create a zero-shot classification pipeline for dynamic demographic assignment
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# Candidate demographic labels (adjust or load these dynamically if needed)
candidate_labels = ["Youth", "Professional", "Senior", "General"]

def get_demographic(token_str):
    token_str_clean = token_str.strip()
    if not token_str_clean:
        return "General"
    # Classify the token into one of the candidate demographic labels
    result = classifier(token_str_clean, candidate_labels)
    return result["labels"][0]  # Highest scoring label

# Example test sentences
test_examples = [
    "<mask>, kya scene hai?",
    "Aaj ka din full <mask> hai.",
    "Project pe <mask> progress chal raha hai.",
    "Meri meeting <mask> ho gayi, totally.",
    "Tum <mask> kaam kar rahe ho, yaar?",
    "Yaar, kal ka schedule <mask> hai, let's plan properly.",
    "Office mein <mask> mood hai aaj, it's really busy.",
    "Dinner ke baad, <mask>movie dekhne ka plan hai?",
    "College ke assignment ke liye <mask> study session ho raha hai.",
    "Bhai, aaj ka game <mask> exciting hai, must watch!"
]

# Process each test example
for example in test_examples:
    print(f"\nInput Sentence: {example}")
    # Get top 5 predictions for the masked token
    predictions = fill_mask(example, top_k=5)
    
    # Dynamically group predicted tokens by inferred demographic
    demographic_results = {}
    for pred in predictions:
        token_str = pred["token_str"].strip()
        demographic = get_demographic(token_str)
        demographic_results.setdefault(demographic, []).append((token_str, pred["score"]))
    
    # Display demographic-specific output
    for demo, tokens in demographic_results.items():
        print(f"Demographic: {demo}")
        for token_str, score in tokens:
            print(f"  Token: '{token_str}' with score: {score:.4f}")
