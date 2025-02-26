from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load your custom fine-tuned model and tokenizer
model_path = "./code_switch_mlm/checkpoint-50"
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
model = AutoModelForMaskedLM.from_pretrained(model_path)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Create a zero-shot classification pipeline using a general-purpose NLI model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define candidate labels for each demographic factor
candidate_labels_age = ["under 30", "over 30"]
candidate_labels_region = ["urban", "rural"]
# candidate_labels_gender = ["male", "female"]

def classify_demographics(token_str):
    """Dynamically classify the predicted token into demographic factors."""
    token_str_clean = token_str.strip()
    if not token_str_clean:
        return {"age": "unknown", "region": "unknown"}
    
    # Classify for age
    result_age = classifier(token_str_clean, candidate_labels_age)
    age_label = result_age["labels"][0]
    
    # Classify for region
    result_region = classifier(token_str_clean, candidate_labels_region)
    region_label = result_region["labels"][0]
    
    # Classify for gender
    # result_gender = classifier(token_str_clean, candidate_labels_gender)
    # gender_label = result_gender["labels"][0]
    
    return {"age": age_label, "region": region_label}

# Example test sentences with a single mask token
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
    # Get the top 5 predictions for the masked token
    predictions = fill_mask(example, top_k=5)
    
    # For each predicted token, dynamically determine the demographics
    for pred in predictions:
        token_str = pred["token_str"].strip()
        demographics = classify_demographics(token_str)
        print(f"Predicted Token: '{token_str}' with score {pred['score']:.4f}")
        print(f"  -> Age: {demographics['age']}, Region: {demographics['region']}")
