import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# ===== Configuration =====
MODEL_PATH = "./code_switch_mlm/checkpoint-50"  # Fine-tuned model path
BASELINE_MODEL_PATH = "xlm-roberta-base"        # Baseline model for comparison

# Create experiment-specific folder for results
EXPERIMENT_NAME = "Experiment_3_Contextual_Regional_Analysis"
RESULTS_DIR = f"results/{EXPERIMENT_NAME}/"
SAVE_RESULTS_PATH = f"{RESULTS_DIR}contextual_regional_analysis_results.csv"
SAVE_FIGS_PATH = f"{RESULTS_DIR}charts/"

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAVE_FIGS_PATH, exist_ok=True)

# ===== Load Models =====
def load_models():
    """Load the fine-tuned model, baseline model, and demographic classifier"""
    print("Loading models...")
    
    # Load tokenizer (same for both models)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    
    # Load fine-tuned model
    model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer, top_k=5)
    
    # Load baseline model
    baseline_model = AutoModelForMaskedLM.from_pretrained(BASELINE_MODEL_PATH)
    baseline_fill_mask = pipeline("fill-mask", model=baseline_model, tokenizer=tokenizer, top_k=5)
    
    # Load demographic classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    
    return tokenizer, fill_mask, baseline_fill_mask, classifier

# ===== Demographic Classification =====
def classify_demographics(classifier, token_str):
    """Classify the demographics of a token using zero-shot classification"""
    token_str_clean = token_str.strip()
    if not token_str_clean:
        return {"age": "unknown", "region": "unknown", "age_confidence": 0, "region_confidence": 0}
    
    # Classify for age
    candidate_labels_age = ["under 30", "over 30"]
    result_age = classifier(token_str_clean, candidate_labels_age)
    age_label = result_age["labels"][0]
    age_confidence = result_age["scores"][0]
    
    # Classify for region
    candidate_labels_region = ["urban", "rural"]
    result_region = classifier(token_str_clean, candidate_labels_region)
    region_label = result_region["labels"][0]
    region_confidence = result_region["scores"][0]
    
    # Classify for formality
    candidate_labels_formality = ["formal", "informal"]
    result_formality = classifier(token_str_clean, candidate_labels_formality)
    formality_label = result_formality["labels"][0]
    formality_confidence = result_formality["scores"][0]
    
    return {
        "age": age_label, 
        "region": region_label,
        "formality": formality_label,
        "age_confidence": age_confidence,
        "region_confidence": region_confidence,
        "formality_confidence": formality_confidence
    }

# ===== Test Cases for Experiment 3 =====
def get_test_cases():
    """
    Create diverse test cases focused on:
    1. Regional variations with specific contextual clues
    2. Context length variations
    3. Script variations (Romanized Hindi vs. English)
    """
    test_cases = {
        "regional_with_context": [
            # North region with strong contextual markers
            {
                "region": "North",
                "context_length": "long",
                "masked_text": "Delhi ki garmi mein <mask> lag raha hai. Red Fort ke paas sabse zyada bheed hai.",
                "expected_tokens": ["pasina", "garam", "hot"],
                "description": "Northern context with landmarks (Delhi, Red Fort)",
                "script_type": "romanized"
            },
            {
                "region": "North",
                "context_length": "short",
                "masked_text": "Delhi mein <mask> khana best hai.",
                "expected_tokens": ["chaat", "street", "local"],
                "description": "Northern context with short reference",
                "script_type": "romanized"
            },
            # South region with strong contextual markers
            {
                "region": "South",
                "context_length": "long",
                "masked_text": "Chennai ke beach par <mask> karte hue bahut maza aaya. Marina Beach is so beautiful in the evening.",
                "expected_tokens": ["walk", "sunset", "time"],
                "description": "Southern context with landmarks (Chennai, Marina Beach)",
                "script_type": "mixed"
            },
            {
                "region": "South",
                "context_length": "short",
                "masked_text": "Bangalore mein <mask> company hai.",
                "expected_tokens": ["IT", "tech", "software"],
                "description": "Southern context with short reference",
                "script_type": "romanized"
            },
            # West region with strong contextual markers
            {
                "region": "West",
                "context_length": "long",
                "masked_text": "Mumbai ke Marine Drive par <mask> dekhna sabse accha lagta hai. Vahan pe sabhi log shaam ko aate hain.",
                "expected_tokens": ["sunset", "sea", "view"],
                "description": "Western context with landmarks (Mumbai, Marine Drive)",
                "script_type": "romanized"
            },
            {
                "region": "West",
                "context_length": "short",
                "masked_text": "Pune mein <mask> weather hai.",
                "expected_tokens": ["accha", "pleasant", "nice"],
                "description": "Western context with short reference",
                "script_type": "romanized"
            }
        ],
        "context_length_variation": [
            # Minimal context
            {
                "region": "Mixed",
                "context_length": "minimal",
                "masked_text": "<mask> khaana.",
                "expected_tokens": ["khana", "dinner", "lunch"],
                "description": "Minimal context with just 1 other word",
                "script_type": "romanized"
            },
            {
                "region": "Mixed",
                "context_length": "minimal",
                "masked_text": "Meeting <mask>.",
                "expected_tokens": ["hai", "khatam", "cancel"],
                "description": "Minimal context with just 1 other word",
                "script_type": "mixed"
            },
            # Short context
            {
                "region": "Mixed",
                "context_length": "short",
                "masked_text": "Office mein <mask> meeting hai.",
                "expected_tokens": ["ek", "important", "team"],
                "description": "Short context with 3-4 words",
                "script_type": "romanized"
            },
            {
                "region": "Mixed",
                "context_length": "short",
                "masked_text": "Main <mask> kaam kar raha hoon.",
                "expected_tokens": ["important", "office", "ghar"],
                "description": "Short context with 5-6 words",
                "script_type": "romanized"
            },
            # Medium context
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "Kal maine market mein shopping karte samay <mask> dekha aur mujhe bahut accha laga.",
                "expected_tokens": ["sale", "discount", "kuch"],
                "description": "Medium context with 10-12 words",
                "script_type": "romanized"
            },
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "Project deadline next week hai, isliye hume <mask> complete karna hoga jaldi se.",
                "expected_tokens": ["work", "documentation", "sab"],
                "description": "Medium context with work setting",
                "script_type": "mixed"
            },
            # Long context
            {
                "region": "Mixed",
                "context_length": "long",
                "masked_text": "Main pichhle hafte se is project par kaam kar raha hoon. Boss ne kaha hai ki agar <mask> time par ho jaye to bonus milega. Isliye main extra hours bhi laga raha hoon.",
                "expected_tokens": ["project", "kaam", "delivery"],
                "description": "Long context with 20+ words across multiple sentences",
                "script_type": "romanized"
            },
            {
                "region": "Mixed",
                "context_length": "long",
                "masked_text": "College ke final year mein humare group ne ek mobile app develop kiya tha. Usme <mask> features the jo users ko bahut pasand aaye. Professor ne bhi kaha tha ki ye commercially viable ho sakta hai.",
                "expected_tokens": ["unique", "interesting", "useful"],
                "description": "Long context with 25+ words across multiple sentences",
                "script_type": "mixed"
            }
        ],
        "script_variation": [
            # Pure Romanized Hindi
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "Aaj maine market mein <mask> kharida kyunki kal dost ke ghar jana hai.",
                "expected_tokens": ["gift", "kuch", "samaan"],
                "description": "Pure Romanized Hindi sentence",
                "script_type": "romanized"
            },
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "Mujhe lagta hai ki iss saal <mask> bahut acchi hogi.",
                "expected_tokens": ["garmi", "barish", "salary"],
                "description": "Pure Romanized Hindi with prediction",
                "script_type": "romanized"
            },
            # Mixed Romanized Hindi and English
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "Weekend par main <mask> movie dekhne jaunga friends ke saath.",
                "expected_tokens": ["new", "latest", "ek"],
                "description": "Mixed Romanized Hindi and English",
                "script_type": "mixed"
            },
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "Meeting ke baad hum <mask> discuss karenge project ke baare mein.",
                "expected_tokens": ["details", "plan", "strategy"],
                "description": "Mixed Romanized Hindi and English in work context",
                "script_type": "mixed"
            },
            # Code-switching at different positions
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "<mask> time par office pahunchna important hai.",
                "expected_tokens": ["on", "right", "correct"],
                "description": "Code-switching at the beginning",
                "script_type": "mixed"
            },
            {
                "region": "Mixed",
                "context_length": "medium",
                "masked_text": "Main ghar <mask> late pahuncha because of traffic.",
                "expected_tokens": ["se", "pe", "mein"],
                "description": "Code-switching in the middle",
                "script_type": "mixed"
            }
        ]
    }
    
    # Flatten the dictionary into a list
    flat_test_cases = []
    for category, cases in test_cases.items():
        for case in cases:
            case["category"] = category
            flat_test_cases.append(case)
    
    return flat_test_cases

# ===== Detect Script Type =====
def detect_script_type(text):
    """
    Detect if a text is in Romanized Hindi, English, or mixed
    """
    # Remove the mask token for analysis
    text = text.replace("<mask>", "")
    
    # Common Hindi words in Romanized form
    hindi_words = ["hai", "hain", "mein", "ka", "ki", "ke", "aur", "par", "kya", "kyun", 
                  "main", "hum", "tum", "aap", "yeh", "woh", "kuch", "bahut", "accha", 
                  "ghar", "kaam", "dost", "pyaar", "zindagi", "samay", "din", "raat"]
    
    # Count Hindi and English word patterns
    hindi_count = 0
    english_count = 0
    
    # Simple tokenization by whitespace
    words = text.lower().split()
    
    for word in words:
        # Clean the word from punctuation
        word = re.sub(r'[^\w\s]', '', word)
        
        if word in hindi_words:
            hindi_count += 1
        # Check if word looks like English (contains only ASCII characters)
        elif all(ord(c) < 128 for c in word) and len(word) > 1:
            # Further check if it's likely an English word and not just a short Hindi word in Roman
            if len(word) > 3 or word in ["is", "am", "are", "the", "of", "and", "to", "in", "by"]:
                english_count += 1
    
    total_words = len(words)
    if total_words == 0:
        return "unknown"
    
    hindi_ratio = hindi_count / total_words
    english_ratio = english_count / total_words
    
    # Determine script type based on ratios
    if hindi_ratio > 0.7:
        return "romanized"
    elif english_ratio > 0.7:
        return "english"
    else:
        return "mixed"

# ===== Evaluate Model =====
def evaluate_model(model_name, fill_mask_pipeline, classifier, test_cases):
    """
    Evaluate model on test cases focused on regional, context length, and script variations
    
    Args:
        model_name: Name of the model being evaluated
        fill_mask_pipeline: The prediction pipeline
        classifier: The demographic classifier
        test_cases: List of test cases with regional and contextual attributes
    
    Returns:
        Dictionary of metrics and DataFrame of results
    """
    print(f"Evaluating {model_name} on contextual and regional patterns...")
    
    results = []
    
    for case in tqdm(test_cases):
        category = case["category"]
        region = case["region"]
        context_length = case["context_length"]
        masked_text = case["masked_text"]
        expected_tokens = case["expected_tokens"]
        description = case["description"]
        script_type = case["script_type"]
        
        # Get predictions
        predictions = fill_mask_pipeline(masked_text)
        
        # Extract predicted tokens and scores
        predicted_tokens = [pred["token_str"] for pred in predictions]
        prediction_scores = [pred["score"] for pred in predictions]
        
        # Check if any expected token is in top-k predictions
        expected_in_top_1 = predicted_tokens[0] in expected_tokens
        expected_in_top_3 = any(token in predicted_tokens[:3] for token in expected_tokens)
        expected_in_top_5 = any(token in predicted_tokens[:5] for token in expected_tokens)
        
        # Find best matching expected token position
        best_match_position = -1
        for token in expected_tokens:
            if token in predicted_tokens:
                position = predicted_tokens.index(token)
                if best_match_position == -1 or position < best_match_position:
                    best_match_position = position
        
        # Get demographic classification for top prediction
        demographics = classify_demographics(classifier, predicted_tokens[0])
        
        # Detect actual script type of the masked text for verification
        detected_script = detect_script_type(masked_text)
        
        # Identify if the prediction is in Hindi or English based on script
        pred_is_english = all(ord(c) < 128 for c in predicted_tokens[0])
        prediction_language = "English" if pred_is_english else "Hindi"
        
        # Store result for this test case
        result = {
            "model": model_name,
            "category": category,
            "region": region,
            "context_length": context_length,
            "masked_text": masked_text,
            "description": description,
            "script_type": script_type,
            "detected_script": detected_script,
            "top_1_prediction": predicted_tokens[0],
            "top_1_score": prediction_scores[0],
            "top_3_predictions": ", ".join(predicted_tokens[:3]),
            "expected_tokens": ", ".join(expected_tokens),
            "expected_in_top_1": expected_in_top_1,
            "expected_in_top_3": expected_in_top_3,
            "expected_in_top_5": expected_in_top_5,
            "best_match_position": best_match_position,
            "predicted_age": demographics["age"],
            "age_confidence": demographics["age_confidence"],
            "predicted_region": demographics["region"],
            "region_confidence": demographics["region_confidence"],
            "predicted_formality": demographics["formality"],
            "formality_confidence": demographics["formality_confidence"],
            "prediction_language": prediction_language
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics by category
    metrics = {}
    
    # Overall metrics
    metrics["overall"] = {
        "model": model_name,
        "cases_evaluated": len(results_df),
        "expected_in_top_1_percent": results_df["expected_in_top_1"].mean() * 100,
        "expected_in_top_3_percent": results_df["expected_in_top_3"].mean() * 100,
        "expected_in_top_5_percent": results_df["expected_in_top_5"].mean() * 100
    }
    
    # Metrics by category
    for category in results_df["category"].unique():
        category_df = results_df[results_df["category"] == category]
        metrics[category] = {
            "cases_evaluated": len(category_df),
            "expected_in_top_1_percent": category_df["expected_in_top_1"].mean() * 100,
            "expected_in_top_3_percent": category_df["expected_in_top_3"].mean() * 100,
            "expected_in_top_5_percent": category_df["expected_in_top_5"].mean() * 100
        }
    
    # Metrics by region
    for region in results_df["region"].unique():
        region_df = results_df[results_df["region"] == region]
        if len(region_df) > 0:
            metrics[f"region_{region}"] = {
                "cases_evaluated": len(region_df),
                "expected_in_top_1_percent": region_df["expected_in_top_1"].mean() * 100,
                "expected_in_top_3_percent": region_df["expected_in_top_3"].mean() * 100,
                "expected_in_top_5_percent": region_df["expected_in_top_5"].mean() * 100
            }
    
    # Metrics by context length
    for context_length in results_df["context_length"].unique():
        context_df = results_df[results_df["context_length"] == context_length]
        if len(context_df) > 0:
            metrics[f"context_{context_length}"] = {
                "cases_evaluated": len(context_df),
                "expected_in_top_1_percent": context_df["expected_in_top_1"].mean() * 100,
                "expected_in_top_3_percent": context_df["expected_in_top_3"].mean() * 100,
                "expected_in_top_5_percent": context_df["expected_in_top_5"].mean() * 100
            }
    
    # Metrics by script type
    for script_type in results_df["script_type"].unique():
        script_df = results_df[results_df["script_type"] == script_type]
        if len(script_df) > 0:
            metrics[f"script_{script_type}"] = {
                "cases_evaluated": len(script_df),
                "expected_in_top_1_percent": script_df["expected_in_top_1"].mean() * 100,
                "expected_in_top_3_percent": script_df["expected_in_top_3"].mean() * 100,
                "expected_in_top_5_percent": script_df["expected_in_top_5"].mean() * 100
            }
    
    return metrics, results_df

# ===== Generate Regional Analysis Visualizations =====
def generate_regional_visualizations(fine_tuned_results, baseline_results):
    """
    Generate visualizations for regional analysis
    """
    print("Generating regional analysis visualizations...")
    
    # Combine results
    fine_tuned_results["model"] = "Fine-tuned Model"
    baseline_results["model"] = "Baseline Model"
    all_results = pd.concat([fine_tuned_results, baseline_results])
    
    # 1. Regional Performance Comparison
    # Get unique regions
    regions = sorted(all_results["region"].unique())
    
    # Prepare data for plotting
    region_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        for region in regions:
            model_region_df = all_results[(all_results["model"] == model) & 
                                        (all_results["region"] == region)]
            
            if len(model_region_df) > 0:
                accuracy_top_1 = model_region_df["expected_in_top_1"].mean() * 100
                
                region_data.append({
                    "Model": model,
                    "Region": region,
                    "Top-1 Accuracy": accuracy_top_1
                })
    
    region_df = pd.DataFrame(region_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Region", y="Top-1 Accuracy", hue="Model", data=region_df)
    
    plt.title("Regional Performance Comparison")
    plt.xlabel("Region")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}regional_performance.png")
    plt.close()
    
    # 2. Context Length Performance Comparison
    # Get unique context lengths
    context_lengths = ["minimal", "short", "medium", "long"]
    
    # Prepare data for plotting
    context_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        for context in context_lengths:
            model_context_df = all_results[(all_results["model"] == model) & 
                                         (all_results["context_length"] == context)]
            
            if len(model_context_df) > 0:
                accuracy_top_1 = model_context_df["expected_in_top_1"].mean() * 100
                
                context_data.append({
                    "Model": model,
                    "Context Length": context,
                    "Top-1 Accuracy": accuracy_top_1
                })
    
    context_df = pd.DataFrame(context_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Context Length", y="Top-1 Accuracy", hue="Model", data=context_df)
    
    plt.title("Context Length Performance Comparison")
    plt.xlabel("Context Length")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}context_length_performance.png")
    plt.close()
    
    # 3. Script Type Performance Comparison
    # Get unique script types
    script_types = sorted(all_results["script_type"].unique())
    
    # Prepare data for plotting
    script_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        for script in script_types:
            model_script_df = all_results[(all_results["model"] == model) & 
                                        (all_results["script_type"] == script)]
            
            if len(model_script_df) > 0:
                accuracy_top_1 = model_script_df["expected_in_top_1"].mean() * 100
                
                script_data.append({
                    "Model": model,
                    "Script Type": script,
                    "Top-1 Accuracy": accuracy_top_1
                })
    
    script_df = pd.DataFrame(script_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Script Type", y="Top-1 Accuracy", hue="Model", data=script_df)
    
    plt.title("Script Type Performance Comparison")
    plt.xlabel("Script Type")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}script_type_performance.png")
    plt.close()
    
    # 4. Language Preference Analysis by Region
    # Analyze if the model chooses Hindi or English based on region
    region_language_data = []
    
    for model in ["Fine-tuned Model", "Baseline Model"]:
        model_df = all_results[all_results["model"] == model]
        
        for region in regions:
            region_df = model_df[model_df["region"] == region]
            
            if len(region_df) > 0:
                english_pct = (region_df["prediction_language"] == "English").mean() * 100
                hindi_pct = 100 - english_pct
                
                region_language_data.append({
                    "Model": model,
                    "Region": region,
                    "English_Percentage": english_pct,
                    "Hindi_Percentage": hindi_pct
                })
    
    region_lang_df = pd.DataFrame(region_language_data)
    
    # Create a figure with multiple subplots for separate bar charts instead of heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Fine-tuned model - bar chart
    ft_data = region_lang_df[region_lang_df["Model"] == "Fine-tuned Model"]
    sns.barplot(x="Region", y="English_Percentage", data=ft_data, ax=axes[0], color="skyblue")
    axes[0].set_title("Regional English Usage (Fine-tuned Model)")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("English Usage (%)")
    
    # Add text labels on the bars
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    # Baseline model - bar chart
    bl_data = region_lang_df[region_lang_df["Model"] == "Baseline Model"]
    sns.barplot(x="Region", y="English_Percentage", data=bl_data, ax=axes[1], color="lightcoral")
    axes[1].set_title("Regional English Usage (Baseline Model)")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("English Usage (%)")
    
    # Add text labels on the bars
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}regional_language_preference.png")
    plt.close()
    
    # 5. Model Improvement by Region
    improvement_data = []
    
    for region in regions:
        ft_region_df = fine_tuned_results[fine_tuned_results["region"] == region]
        bl_region_df = baseline_results[baseline_results["region"] == region]
        
        if len(ft_region_df) > 0 and len(bl_region_df) > 0:
            ft_accuracy = ft_region_df["expected_in_top_1"].mean() * 100
            bl_accuracy = bl_region_df["expected_in_top_1"].mean() * 100
            
            improvement = ft_accuracy - bl_accuracy
            
            improvement_data.append({
                "Region": region,
                "Improvement": improvement
            })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(improvement_df["Region"], improvement_df["Improvement"], color="skyblue")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5 if height >= 0 else height - 2.5,
            f'{height:.1f}%',
            ha='center', 
            va='bottom' if height >= 0 else 'top'
        )
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title("Improvement in Top-1 Accuracy by Region")
    plt.xlabel("Region")
    plt.ylabel("Accuracy Improvement (%)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}regional_improvement.png")
    plt.close()
# ===== Generate Context Analysis Visualizations =====
def generate_context_visualizations(fine_tuned_results, baseline_results):
    """
    Generate visualizations for context length analysis
    """
    print("Generating context length analysis visualizations...")
    
    # Combine results
    fine_tuned_results["model"] = "Fine-tuned Model"
    baseline_results["model"] = "Baseline Model"
    all_results = pd.concat([fine_tuned_results, baseline_results])
    
    # 1. Prediction Confidence by Context Length
    # Get unique context lengths
    context_lengths = ["minimal", "short", "medium", "long"]
    
    # Prepare data for plotting
    confidence_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        for context in context_lengths:
            model_context_df = all_results[(all_results["model"] == model) & 
                                         (all_results["context_length"] == context)]
            
            if len(model_context_df) > 0:
                mean_confidence = model_context_df["top_1_score"].mean()
                
                confidence_data.append({
                    "Model": model,
                    "Context Length": context,
                    "Mean Confidence": mean_confidence
                })
    
    confidence_df = pd.DataFrame(confidence_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Context Length", y="Mean Confidence", hue="Model", data=confidence_df)
    
    plt.title("Prediction Confidence by Context Length")
    plt.xlabel("Context Length")
    plt.ylabel("Mean Confidence Score")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}context_confidence.png")
    plt.close()
    
    # 2. Language Preference by Context Length
    # Analyze if model chooses Hindi or English based on context length
    context_language = []
    
    for model in ["Fine-tuned Model", "Baseline Model"]:
        model_df = all_results[all_results["model"] == model]
        
        for context in context_lengths:
            context_df = model_df[model_df["context_length"] == context]
            
            if len(context_df) > 0:
                english_pct = (context_df["prediction_language"] == "English").mean() * 100
                hindi_pct = 100 - english_pct
                
                context_language.append({
                    "Model": model,
                    "Context Length": context,
                    "English_Percentage": english_pct,
                    "Hindi_Percentage": hindi_pct
                })
    
    context_lang_df = pd.DataFrame(context_language)
    
    # Instead of stacked bar charts, create side-by-side bar charts for better clarity
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Fine-tuned model - bar chart for English percentage
    ft_data = context_lang_df[context_lang_df["Model"] == "Fine-tuned Model"]
    sns.barplot(x="Context Length", y="English_Percentage", data=ft_data, ax=axes[0], color="skyblue")
    axes[0].set_title("English Word Choice by Context Length (Fine-tuned Model)")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("English Usage (%)")
    
    # Add text labels on the bars
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    # Baseline model - bar chart for English percentage
    bl_data = context_lang_df[context_lang_df["Model"] == "Baseline Model"]
    sns.barplot(x="Context Length", y="English_Percentage", data=bl_data, ax=axes[1], color="lightcoral")
    axes[1].set_title("English Word Choice by Context Length (Baseline Model)")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("English Usage (%)")
    
    # Add text labels on the bars
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}context_language_preference.png")
    plt.close()
    
    # 3. Model Improvement by Context Length
    improvement_data = []
    
    for context in context_lengths:
        ft_context_df = fine_tuned_results[fine_tuned_results["context_length"] == context]
        bl_context_df = baseline_results[baseline_results["context_length"] == context]
        
        if len(ft_context_df) > 0 and len(bl_context_df) > 0:
            ft_accuracy = ft_context_df["expected_in_top_1"].mean() * 100
            bl_accuracy = bl_context_df["expected_in_top_1"].mean() * 100
            
            improvement = ft_accuracy - bl_accuracy
            
            improvement_data.append({
                "Context Length": context,
                "Improvement": improvement
            })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(improvement_df["Context Length"], improvement_df["Improvement"], color="skyblue")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5 if height >= 0 else height - 2.5,
            f'{height:.1f}%',
            ha='center', 
            va='bottom' if height >= 0 else 'top'
        )
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title("Improvement in Top-1 Accuracy by Context Length")
    plt.xlabel("Context Length")
    plt.ylabel("Accuracy Improvement (%)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}context_improvement.png")
    plt.close()
# ===== Generate Script Analysis Visualizations =====
def generate_script_visualizations(fine_tuned_results, baseline_results):
    """
    Generate visualizations for script type analysis
    """
    print("Generating script type analysis visualizations...")
    
    # Combine results
    fine_tuned_results["model"] = "Fine-tuned Model"
    baseline_results["model"] = "Baseline Model"
    all_results = pd.concat([fine_tuned_results, baseline_results])
    
    # Get unique script types
    script_types = sorted(all_results["script_type"].unique())
    
    # 1. Prediction Confidence by Script Type
    # Prepare data for plotting
    confidence_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        for script in script_types:
            model_script_df = all_results[(all_results["model"] == model) & 
                                        (all_results["script_type"] == script)]
            
            if len(model_script_df) > 0:
                mean_confidence = model_script_df["top_1_score"].mean()
                
                confidence_data.append({
                    "Model": model,
                    "Script Type": script,
                    "Mean Confidence": mean_confidence
                })
    
    confidence_df = pd.DataFrame(confidence_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="Script Type", y="Mean Confidence", hue="Model", data=confidence_df)
    
    plt.title("Prediction Confidence by Script Type")
    plt.xlabel("Script Type")
    plt.ylabel("Mean Confidence Score")
    plt.ylim(0, 1)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}script_confidence.png")
    plt.close()
    
    # 2. Language Prediction vs Script Type
    # Analyze if model outputs Hindi or English based on input script type
    script_language = []
    
    for model in ["Fine-tuned Model", "Baseline Model"]:
        model_df = all_results[all_results["model"] == model]
        
        for script in script_types:
            script_df = model_df[model_df["script_type"] == script]
            
            if len(script_df) > 0:
                english_pct = (script_df["prediction_language"] == "English").mean() * 100
                hindi_pct = 100 - english_pct
                
                script_language.append({
                    "Model": model,
                    "Script Type": script,
                    "English_Percentage": english_pct,
                    "Hindi_Percentage": hindi_pct
                })
    
    script_lang_df = pd.DataFrame(script_language)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Fine-tuned model - bar chart for English percentage
    ft_data = script_lang_df[script_lang_df["Model"] == "Fine-tuned Model"]
    sns.barplot(x="Script Type", y="English_Percentage", data=ft_data, ax=axes[0], color="skyblue")
    axes[0].set_title("English Word Choice by Script Type (Fine-tuned Model)")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("English Usage (%)")
    
    # Add text labels on the bars
    for p in axes[0].patches:
        axes[0].annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    # Baseline model - bar chart for English percentage
    bl_data = script_lang_df[script_lang_df["Model"] == "Baseline Model"]
    sns.barplot(x="Script Type", y="English_Percentage", data=bl_data, ax=axes[1], color="lightcoral")
    axes[1].set_title("English Word Choice by Script Type (Baseline Model)")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("English Usage (%)")
    
    # Add text labels on the bars
    for p in axes[1].patches:
        axes[1].annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width()/2., p.get_height()), 
                    ha = 'center', va = 'bottom',
                    xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}script_language_preference.png")
    plt.close()
    
    # 3. Model Improvement by Script Type
    improvement_data = []
    
    for script in script_types:
        ft_script_df = fine_tuned_results[fine_tuned_results["script_type"] == script]
        bl_script_df = baseline_results[baseline_results["script_type"] == script]
        
        if len(ft_script_df) > 0 and len(bl_script_df) > 0:
            ft_accuracy = ft_script_df["expected_in_top_1"].mean() * 100
            bl_accuracy = bl_script_df["expected_in_top_1"].mean() * 100
            
            improvement = ft_accuracy - bl_accuracy
            
            improvement_data.append({
                "Script Type": script,
                "Improvement": improvement
            })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(improvement_df["Script Type"], improvement_df["Improvement"], color="skyblue")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5 if height >= 0 else height - 2.5,
            f'{height:.1f}%',
            ha='center', 
            va='bottom' if height >= 0 else 'top'
        )
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title("Improvement in Top-1 Accuracy by Script Type")
    plt.xlabel("Script Type")
    plt.ylabel("Accuracy Improvement (%)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}script_improvement.png")
    plt.close()
# ===== Generate Summary Statistics =====
def generate_summary_statistics(fine_tuned_metrics, baseline_metrics):
    """
    Generate summary statistics from the metrics
    
    Args:
        fine_tuned_metrics: Metrics dictionary for fine-tuned model
        baseline_metrics: Metrics dictionary for baseline model
    """
    print("Generating summary statistics...")
    
    # Create summary dataframe
    summary_rows = []
    
    # Overall performance
    summary_rows.append({
        "Category": "Overall",
        "Fine-tuned Top-1 (%)": fine_tuned_metrics["overall"]["expected_in_top_1_percent"],
        "Baseline Top-1 (%)": baseline_metrics["overall"]["expected_in_top_1_percent"],
        "Improvement (%)": fine_tuned_metrics["overall"]["expected_in_top_1_percent"] - 
                          baseline_metrics["overall"]["expected_in_top_1_percent"]
    })
    
    # Performance by category
    categories = ["regional_with_context", "context_length_variation", "script_variation"]
    for category in categories:
        if category in fine_tuned_metrics and category in baseline_metrics:
            summary_rows.append({
                "Category": category,
                "Fine-tuned Top-1 (%)": fine_tuned_metrics[category]["expected_in_top_1_percent"],
                "Baseline Top-1 (%)": baseline_metrics[category]["expected_in_top_1_percent"],
                "Improvement (%)": fine_tuned_metrics[category]["expected_in_top_1_percent"] - 
                                  baseline_metrics[category]["expected_in_top_1_percent"]
            })
    
    # Performance by region
    regions = ["North", "South", "West", "Mixed"]
    for region in regions:
        region_key = f"region_{region}"
        if region_key in fine_tuned_metrics and region_key in baseline_metrics:
            summary_rows.append({
                "Category": f"Region: {region}",
                "Fine-tuned Top-1 (%)": fine_tuned_metrics[region_key]["expected_in_top_1_percent"],
                "Baseline Top-1 (%)": baseline_metrics[region_key]["expected_in_top_1_percent"],
                "Improvement (%)": fine_tuned_metrics[region_key]["expected_in_top_1_percent"] - 
                                  baseline_metrics[region_key]["expected_in_top_1_percent"]
            })
    
    # Performance by context length
    context_lengths = ["minimal", "short", "medium", "long"]
    for context in context_lengths:
        context_key = f"context_{context}"
        if context_key in fine_tuned_metrics and context_key in baseline_metrics:
            summary_rows.append({
                "Category": f"Context: {context}",
                "Fine-tuned Top-1 (%)": fine_tuned_metrics[context_key]["expected_in_top_1_percent"],
                "Baseline Top-1 (%)": baseline_metrics[context_key]["expected_in_top_1_percent"],
                "Improvement (%)": fine_tuned_metrics[context_key]["expected_in_top_1_percent"] - 
                                  baseline_metrics[context_key]["expected_in_top_1_percent"]
            })
    
    # Performance by script type
    script_types = ["romanized", "mixed"]
    for script in script_types:
        script_key = f"script_{script}"
        if script_key in fine_tuned_metrics and script_key in baseline_metrics:
            summary_rows.append({
                "Category": f"Script: {script}",
                "Fine-tuned Top-1 (%)": fine_tuned_metrics[script_key]["expected_in_top_1_percent"],
                "Baseline Top-1 (%)": baseline_metrics[script_key]["expected_in_top_1_percent"],
                "Improvement (%)": fine_tuned_metrics[script_key]["expected_in_top_1_percent"] - 
                                  baseline_metrics[script_key]["expected_in_top_1_percent"]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary to CSV
    summary_df.to_csv(f"{RESULTS_DIR}summary_statistics.csv", index=False)
    
    return summary_df

# ===== Save All Results to JSON =====
def save_experiment_results(fine_tuned_metrics, baseline_metrics):
    """
    Save all experiment results in structured JSON format
    
    Args:
        fine_tuned_metrics: Metrics dictionary for fine-tuned model
        baseline_metrics: Metrics dictionary for baseline model
    """
    print("Saving experiment results to JSON...")
    
    # Create structured results dictionary
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fine_tuned_model": {
            "path": MODEL_PATH,
            "metrics": fine_tuned_metrics
        },
        "baseline_model": {
            "path": BASELINE_MODEL_PATH,
            "metrics": baseline_metrics
        },
        "experiment_summary": {
            "overall_improvement": fine_tuned_metrics["overall"]["expected_in_top_1_percent"] - 
                                  baseline_metrics["overall"]["expected_in_top_1_percent"]
        }
    }
    
    # Save to JSON file
    with open(f"{RESULTS_DIR}experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)

# ===== Main Function =====
def main():
    """Main experiment function"""
    # Print experiment information
    print(f"Running {EXPERIMENT_NAME}")
    print(f"Results will be saved to {RESULTS_DIR}")
    
    # Load models
    tokenizer, fine_tuned_model, baseline_model, classifier = load_models()
    
    # Get test cases
    test_cases = get_test_cases()
    
    # Evaluate fine-tuned model
    fine_tuned_metrics, fine_tuned_results = evaluate_model(
        "Fine-tuned Model", fine_tuned_model, classifier, test_cases
    )
    
    # Evaluate baseline model
    baseline_metrics, baseline_results = evaluate_model(
        "Baseline Model", baseline_model, classifier, test_cases
    )
    
    # Combine results
    all_results = pd.concat([fine_tuned_results, baseline_results])
    all_results.to_csv(SAVE_RESULTS_PATH, index=False)
    
    # Generate visualizations
    generate_regional_visualizations(fine_tuned_results, baseline_results)
    generate_context_visualizations(fine_tuned_results, baseline_results)
    generate_script_visualizations(fine_tuned_results, baseline_results)
    
    # Generate summary statistics
    summary_df = generate_summary_statistics(fine_tuned_metrics, baseline_metrics)
    
    # Save experiment results
    save_experiment_results(fine_tuned_metrics, baseline_metrics)
    
    # Print summary results
    print("\n===== Summary Results =====")
    print(summary_df.to_string(index=False))
    
    # Print detailed category performance
    print("\n===== Detailed Category Performance =====")
    
    # By category
    categories = ["regional_with_context", "context_length_variation", "script_variation"]
    for category in categories:
        if category in fine_tuned_metrics and category in baseline_metrics:
            print(f"\n{category.upper().replace('_', ' ')}:")
            print(f"  Fine-tuned: Top-1: {fine_tuned_metrics[category]['expected_in_top_1_percent']:.1f}%, "
                  f"Top-3: {fine_tuned_metrics[category]['expected_in_top_3_percent']:.1f}%")
            print(f"  Baseline:   Top-1: {baseline_metrics[category]['expected_in_top_1_percent']:.1f}%, "
                  f"Top-3: {baseline_metrics[category]['expected_in_top_3_percent']:.1f}%")
            improvement = fine_tuned_metrics[category]['expected_in_top_1_percent'] - baseline_metrics[category]['expected_in_top_1_percent']
            print(f"  Improvement: {improvement:.1f}%")
    
    # Print specific examples
    print("\n===== Sample Predictions =====")
    # Get one example from each category
    for category in categories:
        category_examples = fine_tuned_results[fine_tuned_results["category"] == category]
        
        if len(category_examples) > 0:
            example = category_examples.iloc[0]
            
            print(f"\nCategory: {category}")
            print(f"Region: {example['region']}, Context Length: {example['context_length']}, Script Type: {example['script_type']}")
            print(f"Masked text: {example['masked_text']}")
            print(f"Fine-tuned model prediction: '{example['top_1_prediction']}' (Score: {example['top_1_score']:.4f})")
            
            # Get corresponding baseline prediction
            baseline_example = baseline_results[
                (baseline_results["category"] == category) & 
                (baseline_results["masked_text"] == example["masked_text"])
            ]
            
            if len(baseline_example) > 0:
                baseline_pred = baseline_example.iloc[0]
                print(f"Baseline model prediction: '{baseline_pred['top_1_prediction']}' (Score: {baseline_pred['top_1_score']:.4f})")
            
            print(f"Expected tokens: {example['expected_tokens']}")
    
    print(f"\nAll results saved to {RESULTS_DIR}")
    print(f"Experiment completed successfully!")

if __name__ == "__main__":
    main()