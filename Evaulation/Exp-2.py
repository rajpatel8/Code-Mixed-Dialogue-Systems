import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# ===== Configuration =====
MODEL_PATH = "./code_switch_mlm/checkpoint-50"  # Fine-tuned model path
BASELINE_MODEL_PATH = "xlm-roberta-base"        # Baseline model for comparison

# Create experiment-specific folder for results
EXPERIMENT_NAME = "Experiment_2_Demographic_Analysis"
RESULTS_DIR = f"results/{EXPERIMENT_NAME}/"
SAVE_RESULTS_PATH = f"{RESULTS_DIR}demographic_analysis_results.csv"
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

# ===== Demographic-Specific Test Cases =====
def get_demographic_test_cases():
    """
    Create test cases specifically designed to test demographic sensitivity
    Each case has demographic attributes and expected tokens
    """
    test_cases = {
        "age_specific": [
            {
                "demographic": "Teen",
                "masked_text": "Bhai, <mask> ho raha hai?",
                "expected_tokens": ["kya", "time", "game"],
                "description": "Teen greeting pattern"
            },
            {
                "demographic": "Teen",
                "masked_text": "<mask> movie dekhne chale weekend pe?",
                "expected_tokens": ["kaunsi", "new", "latest"],
                "description": "Teen entertainment context"
            },
            {
                "demographic": "Adult",
                "masked_text": "Meeting ke <mask> presentation ready karni hai.",
                "expected_tokens": ["liye", "baad", "pehle"],
                "description": "Adult work context"
            },
            {
                "demographic": "Adult",
                "masked_text": "<mask>, aapka presentation kaisa chal raha hai?",
                "expected_tokens": ["Sir", "Ma'am", "Hello"],
                "description": "Adult formal greeting"
            },
            {
                "demographic": "Senior",
                "masked_text": "<mask>, aap kaise hain?",
                "expected_tokens": ["Namaste", "Sir", "Hello"],
                "description": "Senior formal greeting"
            },
            {
                "demographic": "Senior",
                "masked_text": "Main <mask> baat kar raha hoon.",
                "expected_tokens": ["se", "phone", "Hindi"],
                "description": "Senior communication style"
            }
        ],
        "region_specific": [
            {
                "demographic": "North",
                "masked_text": "Yaar, <mask> kaam ho gaya?",
                "expected_tokens": ["tera", "tumhara", "homework"],
                "description": "Northern dialectal pattern"
            },
            {
                "demographic": "North",
                "masked_text": "Aaj main <mask> market jaunga.",
                "expected_tokens": ["local", "Sadar", "Connaught"],
                "description": "Northern location reference"
            },
            {
                "demographic": "South",
                "masked_text": "Project ka <mask> complete ho gaya?",
                "expected_tokens": ["work", "module", "planning"],
                "description": "Southern work context pattern"
            },
            {
                "demographic": "South",
                "masked_text": "<mask> dinner karenge aaj?",
                "expected_tokens": ["Kahan", "Where", "Restaurant"],
                "description": "Southern social context"
            },
            {
                "demographic": "West",
                "masked_text": "Meeting <mask> schedule kar di hai.",
                "expected_tokens": ["time", "ka", "ki"],
                "description": "Western formal context"
            },
            {
                "demographic": "West",
                "masked_text": "<mask> message bhej dena details ke saath.",
                "expected_tokens": ["WhatsApp", "email", "ek"],
                "description": "Western communication pattern"
            }
        ],
        "formality_specific": [
            {
                "demographic": "Formal",
                "masked_text": "<mask>, kya aap meeting ke liye taiyar hain?",
                "expected_tokens": ["Sir", "Ma'am", "Hello"],
                "description": "Formal workplace greeting"
            },
            {
                "demographic": "Formal",
                "masked_text": "Project ki <mask> report submit kar di hai.",
                "expected_tokens": ["final", "detailed", "progress"],
                "description": "Formal work context"
            },
            {
                "demographic": "Informal",
                "masked_text": "<mask>, kya scene hai?",
                "expected_tokens": ["Bhai", "Yaar", "Hey"],
                "description": "Informal greeting"
            },
            {
                "demographic": "Informal",
                "masked_text": "Movie <mask> mast thi, dekhni chahiye tujhe.",
                "expected_tokens": ["bahut", "ekdum", "totally"],
                "description": "Informal entertainment context"
            }
        ],
        "code_switching_boundaries": [
            {
                "demographic": "Mixed",
                "masked_text": "Main kal <mask> movie dekhne gaya tha.",
                "expected_tokens": ["ek", "new", "Hollywood"],
                "description": "Hindi to English boundary"
            },
            {
                "demographic": "Mixed",
                "masked_text": "The meeting <mask> bahut important hai.",
                "expected_tokens": ["is", "was", "agenda"],
                "description": "English to Hindi boundary"
            },
            {
                "demographic": "Mixed",
                "masked_text": "Project deadline <mask> extend karna padega.",
                "expected_tokens": ["ko", "se", "tak"],
                "description": "Complex boundary with English terms in Hindi structure"
            },
            {
                "demographic": "Mixed",
                "masked_text": "Can you please <mask> bata do?",
                "expected_tokens": ["details", "time", "schedule"],
                "description": "English to Hindi transition"
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

# ===== Evaluate Demographic Sensitivity =====
def evaluate_demographic_sensitivity(model_name, fill_mask_pipeline, classifier, test_cases):
    """
    Evaluate how well the model predicts tokens according to demographic expectations
    
    Args:
        model_name: Name of the model being evaluated
        fill_mask_pipeline: The prediction pipeline
        classifier: The demographic classifier
        test_cases: List of test cases with demographic attributes
    
    Returns:
        Dictionary of results with demographic analysis
    """
    print(f"Evaluating demographic sensitivity for {model_name}...")
    
    results = []
    
    for case in tqdm(test_cases):
        category = case["category"]
        demographic = case["demographic"]
        masked_text = case["masked_text"]
        expected_tokens = case["expected_tokens"]
        description = case["description"]
        
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
        
        # Store result for this test case
        result = {
            "model": model_name,
            "category": category,
            "demographic": demographic,
            "masked_text": masked_text,
            "description": description,
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
            "formality_confidence": demographics["formality_confidence"]
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate metrics by demographic category
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
    
    # Metrics by specific demographic
    for demographic in results_df["demographic"].unique():
        demographic_df = results_df[results_df["demographic"] == demographic]
        if len(demographic_df) > 0:
            metrics[f"demographic_{demographic}"] = {
                "cases_evaluated": len(demographic_df),
                "expected_in_top_1_percent": demographic_df["expected_in_top_1"].mean() * 100,
                "expected_in_top_3_percent": demographic_df["expected_in_top_3"].mean() * 100,
                "expected_in_top_5_percent": demographic_df["expected_in_top_5"].mean() * 100
            }
    
    return metrics, results_df

# ===== Generate Demographic Analysis Visualizations =====
def generate_demographic_visualizations(fine_tuned_results, baseline_results):
    """
    Generate visualizations showing demographic analysis results
    
    Args:
        fine_tuned_results: Results DataFrame for fine-tuned model
        baseline_results: Results DataFrame for baseline model
    """
    print("Generating demographic analysis visualizations...")
    
    # Combine results for easier comparison
    fine_tuned_results["model"] = "Fine-tuned Model"
    baseline_results["model"] = "Baseline Model"
    all_results = pd.concat([fine_tuned_results, baseline_results])
    
    # 1. Performance by demographic category
    plt.figure(figsize=(12, 8))
    
    # Get unique categories
    categories = sorted(all_results["category"].unique())
    
    # Prepare data for plotting
    category_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        for category in categories:
            model_category_df = all_results[(all_results["model"] == model) & 
                                          (all_results["category"] == category)]
            
            accuracy_top_1 = model_category_df["expected_in_top_1"].mean() * 100
            accuracy_top_3 = model_category_df["expected_in_top_3"].mean() * 100
            
            category_data.append({
                "Model": model,
                "Category": category,
                "Metric": "Top-1 Accuracy",
                "Value": accuracy_top_1
            })
            
            category_data.append({
                "Model": model,
                "Category": category,
                "Metric": "Top-3 Accuracy",
                "Value": accuracy_top_3
            })
    
    category_df = pd.DataFrame(category_data)
    
    # Plot
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Category", y="Value", hue="Model", data=category_df[category_df["Metric"] == "Top-1 Accuracy"])
    
    plt.title("Top-1 Accuracy by Demographic Category")
    plt.xlabel("Demographic Category")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}category_accuracy.png")
    plt.close()
    
    # 2. Performance by specific demographic
    plt.figure(figsize=(14, 8))
    
    # Get unique demographics
    demographics = sorted(all_results["demographic"].unique())
    
    # Prepare data for plotting
    demographic_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        for demographic in demographics:
            model_demographic_df = all_results[(all_results["model"] == model) & 
                                             (all_results["demographic"] == demographic)]
            
            if len(model_demographic_df) > 0:
                accuracy_top_1 = model_demographic_df["expected_in_top_1"].mean() * 100
                
                demographic_data.append({
                    "Model": model,
                    "Demographic": demographic,
                    "Value": accuracy_top_1
                })
    
    demographic_df = pd.DataFrame(demographic_data)
    
    # Plot
    plt.figure(figsize=(16, 8))
    ax = sns.barplot(x="Demographic", y="Value", hue="Model", data=demographic_df)
    
    plt.title("Top-1 Accuracy by Specific Demographic")
    plt.xlabel("Demographic")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}demographic_accuracy.png")
    plt.close()
    
    # 3. Code-switching boundary accuracy
    plt.figure(figsize=(12, 6))
    
    # Filter for code-switching boundary cases
    boundary_results = all_results[all_results["category"] == "code_switching_boundaries"]
    
    # Prepare data for plotting
    boundary_data = []
    for model in ["Fine-tuned Model", "Baseline Model"]:
        model_boundary_df = boundary_results[boundary_results["model"] == model]
        
        for _, row in model_boundary_df.iterrows():
            boundary_data.append({
                "Model": model,
                "Boundary Type": row["description"],
                "Accuracy": 1 if row["expected_in_top_3"] else 0,
                "Confidence": row["top_1_score"]
            })
    
    boundary_df = pd.DataFrame(boundary_data)
    
    # Plot confidence by boundary type
    plt.figure(figsize=(14, 6))
    ax = sns.barplot(x="Boundary Type", y="Confidence", hue="Model", data=boundary_df)
    
    plt.title("Prediction Confidence at Code-Switching Boundaries")
    plt.xlabel("Boundary Type")
    plt.ylabel("Confidence Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}boundary_confidence.png")
    plt.close()
    
    # 4. Demographic classification distribution
    formality_by_demographic = pd.crosstab(
        all_results[all_results["model"] == "Fine-tuned Model"]["demographic"],
        all_results[all_results["model"] == "Fine-tuned Model"]["predicted_formality"],
        normalize="index"
    ) * 100
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(formality_by_demographic, annot=True, cmap="YlGnBu", fmt=".1f", 
                cbar_kws={'label': 'Percentage (%)'})
    plt.title("Formality Classification by Demographic (Fine-tuned Model)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}formality_by_demographic.png")
    plt.close()
    
    # 5. Model improvement across categories
    improvement_data = []
    
    for category in categories:
        ft_df = fine_tuned_results[fine_tuned_results["category"] == category]
        bl_df = baseline_results[baseline_results["category"] == category]
        
        ft_accuracy = ft_df["expected_in_top_1"].mean() * 100
        bl_accuracy = bl_df["expected_in_top_1"].mean() * 100
        
        improvement = ft_accuracy - bl_accuracy
        
        improvement_data.append({
            "Category": category,
            "Improvement": improvement
        })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Plot improvement
    plt.figure(figsize=(12, 6))
    bars = plt.bar(improvement_df["Category"], improvement_df["Improvement"], color="skyblue")
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5,
            f'{height:.1f}%',
            ha='center', 
            va='bottom'
        )
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title("Improvement in Top-1 Accuracy from Fine-tuning")
    plt.xlabel("Category")
    plt.ylabel("Accuracy Improvement (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}model_improvement.png")
    plt.close()

# ===== Analyze Code-Switching Patterns =====
def analyze_code_switching_patterns(all_results):
    """
    Analyze patterns in code-switching behavior across demographics
    
    Args:
        all_results: Combined results DataFrame
    """
    print("Analyzing code-switching patterns...")
    
    # Create directory for output
    os.makedirs(f"{SAVE_FIGS_PATH}patterns/", exist_ok=True)
    
    # Filter for fine-tuned model results
    ft_results = all_results[all_results["model"] == "Fine-tuned Model"]
    
    # 1. Analyze language preference by demographic
    # Identify if prediction is Hindi or English based on script
    def is_english(text):
        # Simple heuristic: if all characters are ASCII, assume English
        return all(ord(c) < 128 for c in text)
    
    # Add language identification
    ft_results["predicted_language"] = ft_results["top_1_prediction"].apply(
        lambda x: "English" if is_english(x) else "Hindi"
    )
    
    # Calculate language preference by demographic
    language_by_demographic = pd.crosstab(
        ft_results["demographic"],
        ft_results["predicted_language"],
        normalize="index"
    ) * 100
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.heatmap(language_by_demographic, annot=True, cmap="coolwarm", fmt=".1f",
                cbar_kws={'label': 'Percentage (%)'})
    plt.title("Language Preference by Demographic")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}patterns/language_preference.png")
    plt.close()
    
    # 2. Analyze language preference by formality
    language_by_formality = pd.crosstab(
        ft_results["predicted_formality"],
        ft_results["predicted_language"],
        normalize="index"
    ) * 100
    
    plt.figure(figsize=(8, 5))
    sns.heatmap(language_by_formality, annot=True, cmap="coolwarm", fmt=".1f",
                cbar_kws={'label': 'Percentage (%)'})
    plt.title("Language Preference by Formality")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}patterns/language_by_formality.png")
    plt.close()
    
    # 3. Analyze prediction confidence by demographic and language
    confidence_data = []
    
    for demographic in ft_results["demographic"].unique():
        for language in ["English", "Hindi"]:
            subset = ft_results[(ft_results["demographic"] == demographic) & 
                               (ft_results["predicted_language"] == language)]
            
            if len(subset) > 0:
                mean_confidence = subset["top_1_score"].mean()
                
                confidence_data.append({
                    "Demographic": demographic,
                    "Language": language,
                    "Mean Confidence": mean_confidence
                })
    
    confidence_df = pd.DataFrame(confidence_data)
    
    # Pivot for heatmap
    confidence_pivot = confidence_df.pivot(
        index="Demographic", 
        columns="Language", 
        values="Mean Confidence"
    )
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confidence_pivot, annot=True, cmap="YlGnBu", fmt=".3f",
                cbar_kws={'label': 'Mean Confidence Score'})
    plt.title("Prediction Confidence by Demographic and Language")
    plt.tight_layout()
    plt.savefig(f"{SAVE_FIGS_PATH}patterns/confidence_by_demographic_language.png")
    plt.close()
    
    # Return pattern analysis
    return {
        "language_by_demographic": language_by_demographic,
        "language_by_formality": language_by_formality,
        "confidence_by_demographic_language": confidence_pivot
    }

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
    for category in ["age_specific", "region_specific", "formality_specific", "code_switching_boundaries"]:
        if category in fine_tuned_metrics and category in baseline_metrics:
            summary_rows.append({
                "Category": category,
                "Fine-tuned Top-1 (%)": fine_tuned_metrics[category]["expected_in_top_1_percent"],
                "Baseline Top-1 (%)": baseline_metrics[category]["expected_in_top_1_percent"],
                "Improvement (%)": fine_tuned_metrics[category]["expected_in_top_1_percent"] - 
                                  baseline_metrics[category]["expected_in_top_1_percent"]
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save summary to CSV
    summary_df.to_csv(f"{RESULTS_DIR}summary_statistics.csv", index=False)
    
    return summary_df

# ===== Save All Results to JSON =====
def save_experiment_results(fine_tuned_metrics, baseline_metrics, pattern_analysis):
    """
    Save all experiment results in structured JSON format
    
    Args:
        fine_tuned_metrics: Metrics dictionary for fine-tuned model
        baseline_metrics: Metrics dictionary for baseline model
        pattern_analysis: Analysis of code-switching patterns
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
        "demographic_analysis": {
            "language_preference_by_demographic": pattern_analysis["language_by_demographic"].to_dict(),
            "language_preference_by_formality": pattern_analysis["language_by_formality"].to_dict()
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
    
    # Get demographic test cases
    test_cases = get_demographic_test_cases()
    
    # Evaluate fine-tuned model
    fine_tuned_metrics, fine_tuned_results = evaluate_demographic_sensitivity(
        "Fine-tuned Model", fine_tuned_model, classifier, test_cases
    )
    
    # Evaluate baseline model
    baseline_metrics, baseline_results = evaluate_demographic_sensitivity(
        "Baseline Model", baseline_model, classifier, test_cases
    )
    
    # Combine results
    all_results = pd.concat([fine_tuned_results, baseline_results])
    all_results.to_csv(SAVE_RESULTS_PATH, index=False)
    
    # Generate visualizations
    generate_demographic_visualizations(fine_tuned_results, baseline_results)
    
    # Analyze code-switching patterns
    pattern_analysis = analyze_code_switching_patterns(all_results)
    
    # Generate summary statistics
    summary_df = generate_summary_statistics(fine_tuned_metrics, baseline_metrics)
    
    # Save experiment results
    save_experiment_results(fine_tuned_metrics, baseline_metrics, pattern_analysis)
    
    # Print summary results
    print("\n===== Summary Results =====")
    print(summary_df.to_string(index=False))
    
    # Print detailed category performance
    print("\n===== Detailed Category Performance =====")
    for category in ["age_specific", "region_specific", "formality_specific", "code_switching_boundaries"]:
        print(f"\n{category.upper()}:")
        print(f"  Fine-tuned: Top-1: {fine_tuned_metrics[category]['expected_in_top_1_percent']:.1f}%, "
              f"Top-3: {fine_tuned_metrics[category]['expected_in_top_3_percent']:.1f}%")
        print(f"  Baseline:   Top-1: {baseline_metrics[category]['expected_in_top_1_percent']:.1f}%, "
              f"Top-3: {baseline_metrics[category]['expected_in_top_3_percent']:.1f}%")
        print(f"  Improvement: {fine_tuned_metrics[category]['expected_in_top_1_percent'] - baseline_metrics[category]['expected_in_top_1_percent']:.1f}%")
    
    # Print specific examples
    print("\n===== Sample Predictions =====")
    # Get one example from each category
    categories = ["age_specific", "region_specific", "formality_specific", "code_switching_boundaries"]
    for category in categories:
        category_examples = fine_tuned_results[fine_tuned_results["category"] == category]
        
        if len(category_examples) > 0:
            example = category_examples.iloc[0]
            
            print(f"\nCategory: {category}")
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
            print(f"Demographics: Age: {example['predicted_age']}, Region: {example['predicted_region']}, Formality: {example['predicted_formality']}")
    
    print(f"\nAll results saved to {RESULTS_DIR}")
    print(f"Experiment completed successfully!")

if __name__ == "__main__":
    main()