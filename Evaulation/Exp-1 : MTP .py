import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# ===== Configuration =====
MODEL_PATH = "./code_switch_mlm/checkpoint-50"  # Your model path from your script
BASELINE_MODEL_PATH = "xlm-roberta-base"        # Baseline model for comparison
SAVE_RESULTS_PATH = "token_prediction_results.csv"
SAVE_FIGS_PATH = "token_prediction_charts/"

# ===== Load Models =====
def load_models():
    """Load both the fine-tuned model and baseline model"""
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
    """
    Classify the demographics of a token using zero-shot classification
    - Adapted from your original function
    """
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
    
    return {
        "age": age_label, 
        "region": region_label,
        "age_confidence": age_confidence,
        "region_confidence": region_confidence
    }

# ===== Test Samples =====
def get_test_samples():
    """
    Returns the test examples from your original script
    """
    # Use your original test examples
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
        "Bhai, aaj ka game <mask> exciting hai, must watch!",
        "<mask> was working on this project.",
        "<mask> what are you doing?"
    ]
    
    # Convert to a standardized format
    formatted_samples = []
    for example in test_examples:
        # We don't have ground truth for these examples, so we'll use the model's top prediction
        # as a placeholder for evaluation purposes
        formatted_samples.append({
            "masked_text": example,
            "correct_token": None,  # Will be filled in after initial prediction
            "demographics": {
                "age": None,  # Will be determined from the predicted token
                "gender": None,  # Not used in this evaluation
                "region": None  # Will be determined from the predicted token
            }
        })
    
    return formatted_samples

# ===== Prepare Ground Truth =====
def prepare_ground_truth(fill_mask_pipeline, classifier, test_samples):
    """
    Since we don't have manually annotated ground truth, we'll use the model's 
    top prediction as reference for evaluation purposes
    """
    print("Preparing pseudo ground-truth data...")
    
    for sample in test_samples:
        # Get top prediction from fine-tuned model
        predictions = fill_mask_pipeline(sample["masked_text"])
        top_token = predictions[0]["token_str"]
        
        # Set as pseudo ground truth
        sample["correct_token"] = top_token
        
        # Determine demographics
        demographics = classify_demographics(classifier, top_token)
        sample["demographics"]["age"] = demographics["age"]
        sample["demographics"]["region"] = demographics["region"]
    
    return test_samples

# ===== Evaluate Prediction =====
def evaluate_predictions(model_name, fill_mask_pipeline, classifier, test_samples):
    """
    Evaluate the model's prediction performance on test samples
    
    Args:
        model_name: Name of the model being evaluated
        fill_mask_pipeline: The prediction pipeline
        classifier: The demographic classifier
        test_samples: List of test sample dictionaries
    
    Returns:
        Dictionary of evaluation metrics and DataFrame of results
    """
    print(f"Evaluating {model_name}...")
    
    results = []
    
    for sample in tqdm(test_samples):
        masked_text = sample["masked_text"]
        correct_token = sample["correct_token"]
        
        # Get predictions
        predictions = fill_mask_pipeline(masked_text)
        
        # Extract predicted tokens and scores
        predicted_tokens = [pred["token_str"] for pred in predictions]
        prediction_scores = [pred["score"] for pred in predictions]
        
        # Get demographic classification for top prediction
        demographics = classify_demographics(classifier, predicted_tokens[0])
        
        # Calculate metrics
        is_top_1 = correct_token == predicted_tokens[0]
        is_top_3 = correct_token in predicted_tokens[:3]
        is_top_5 = correct_token in predicted_tokens[:5]
        
        # Find position of correct token if present
        try:
            correct_rank = predicted_tokens.index(correct_token) + 1
            reciprocal_rank = 1.0 / correct_rank
        except ValueError:
            correct_rank = 0
            reciprocal_rank = 0.0
        
        # Store result for this sample
        result = {
            "model": model_name,
            "masked_text": masked_text,
            "correct_token": correct_token,
            "top_1_prediction": predicted_tokens[0],
            "top_1_score": prediction_scores[0],
            "top_2_prediction": predicted_tokens[1] if len(predicted_tokens) > 1 else "",
            "top_2_score": prediction_scores[1] if len(prediction_scores) > 1 else 0,
            "top_3_prediction": predicted_tokens[2] if len(predicted_tokens) > 2 else "",
            "top_3_score": prediction_scores[2] if len(prediction_scores) > 2 else 0,
            "is_correct_top_1": is_top_1,
            "is_correct_top_3": is_top_3,
            "is_correct_top_5": is_top_5,
            "correct_rank": correct_rank,
            "reciprocal_rank": reciprocal_rank,
            "predicted_age": demographics["age"],
            "age_confidence": demographics["age_confidence"],
            "predicted_region": demographics["region"],
            "region_confidence": demographics["region_confidence"],
            "ground_truth_age": sample["demographics"]["age"],
            "ground_truth_region": sample["demographics"]["region"]
        }
        
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    metrics = {
        "model": model_name,
        "top_1_accuracy": results_df["is_correct_top_1"].mean() if "is_correct_top_1" in results_df.columns else None,
        "top_3_accuracy": results_df["is_correct_top_3"].mean() if "is_correct_top_3" in results_df.columns else None,
        "top_5_accuracy": results_df["is_correct_top_5"].mean() if "is_correct_top_5" in results_df.columns else None,
        "mrr": results_df["reciprocal_rank"].mean() if "reciprocal_rank" in results_df.columns else None,
        "samples_evaluated": len(results_df)
    }
    
    # Calculate age classification accuracy
    if "predicted_age" in results_df.columns and "ground_truth_age" in results_df.columns:
        age_matches = results_df["predicted_age"] == results_df["ground_truth_age"]
        metrics["age_classification_accuracy"] = age_matches.mean()
    
    # Calculate region classification accuracy
    if "predicted_region" in results_df.columns and "ground_truth_region" in results_df.columns:
        region_matches = results_df["predicted_region"] == results_df["ground_truth_region"]
        metrics["region_classification_accuracy"] = region_matches.mean()
    
    return metrics, results_df

# ===== Analyze Token Distributions =====
def analyze_token_distributions(all_results_df):
    """
    Analyze the distribution of predicted tokens across demographic groups
    """
    print("Analyzing token distributions...")
    
    # Get unique tokens and demographic attributes
    top_tokens = {}
    for model in all_results_df["model"].unique():
        model_df = all_results_df[all_results_df["model"] == model]
        
        # Get top 10 most frequent tokens
        token_counts = model_df["top_1_prediction"].value_counts().head(10)
        top_tokens[model] = token_counts.index.tolist()
    
    # Analyze age distribution for top tokens
    age_distributions = {}
    for model in all_results_df["model"].unique():
        model_df = all_results_df[all_results_df["model"] == model]
        
        model_age_dist = {}
        for token in top_tokens[model]:
            token_rows = model_df[model_df["top_1_prediction"] == token]
            if len(token_rows) > 0:
                under_30_pct = (token_rows["predicted_age"] == "under 30").mean()
                model_age_dist[token] = {"under_30": under_30_pct, "over_30": 1 - under_30_pct}
        
        age_distributions[model] = model_age_dist
    
    # Analyze region distribution for top tokens
    region_distributions = {}
    for model in all_results_df["model"].unique():
        model_df = all_results_df[all_results_df["model"] == model]
        
        model_region_dist = {}
        for token in top_tokens[model]:
            token_rows = model_df[model_df["top_1_prediction"] == token]
            if len(token_rows) > 0:
                urban_pct = (token_rows["predicted_region"] == "urban").mean()
                model_region_dist[token] = {"urban": urban_pct, "rural": 1 - urban_pct}
        
        region_distributions[model] = model_region_dist
    
    return {
        "top_tokens": top_tokens,
        "age_distributions": age_distributions,
        "region_distributions": region_distributions
    }

# ===== Generate Token Distribution Visualizations =====
def generate_token_distributions_viz(distribution_analysis):
    """
    Generate visualizations of token distributions across demographic groups
    """
    print("Generating token distribution visualizations...")
    
    # Create directory for figures if it doesn't exist
    import os
    os.makedirs(SAVE_FIGS_PATH, exist_ok=True)
    
    # Plot age distributions for fine-tuned model
    model = "Fine-tuned Code-Switch Model"
    if model in distribution_analysis["top_tokens"]:
        # Age distribution heatmap
        tokens = distribution_analysis["top_tokens"][model]
        
        # Prepare data for heatmap
        age_data = []
        for token in tokens:
            if token in distribution_analysis["age_distributions"][model]:
                age_dist = distribution_analysis["age_distributions"][model][token]
                age_data.append([
                    age_dist["under_30"],
                    age_dist["over_30"]
                ])
        
        if len(age_data) > 0:
            age_df = pd.DataFrame(age_data, index=tokens, columns=["under 30", "over 30"])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(age_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Proportion'})
            plt.title(f'Age Distribution for Top Tokens ({model})')
            plt.tight_layout()
            plt.savefig(f"{SAVE_FIGS_PATH}age_distribution_heatmap.png")
            plt.close()
        
        # Region distribution heatmap
        region_data = []
        for token in tokens:
            if token in distribution_analysis["region_distributions"][model]:
                region_dist = distribution_analysis["region_distributions"][model][token]
                region_data.append([
                    region_dist["urban"],
                    region_dist["rural"]
                ])
        
        if len(region_data) > 0:
            region_df = pd.DataFrame(region_data, index=tokens, columns=["urban", "rural"])
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(region_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar_kws={'label': 'Proportion'})
            plt.title(f'Region Distribution for Top Tokens ({model})')
            plt.tight_layout()
            plt.savefig(f"{SAVE_FIGS_PATH}region_distribution_heatmap.png")
            plt.close()

# ===== Generate Prediction Tables =====
def generate_prediction_tables(all_results_df):
    """
    Generate detailed prediction tables for analysis
    -- Fixed version to avoid aggregation errors with string data
    """
    print("Generating prediction tables...")
    
    # Create summary table of all predictions
    summary_rows = []
    
    for idx, row in all_results_df.iterrows():
        summary_rows.append({
            "Masked Text": row["masked_text"],
            "Model": row["model"],
            "Top-1 Prediction": row["top_1_prediction"],
            "Score": row["top_1_score"],
            "Predicted Age": row["predicted_age"],
            "Age Confidence": row["age_confidence"],
            "Predicted Region": row["predicted_region"],
            "Region Confidence": row["region_confidence"]
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Instead of using pivot_table which attempts aggregation,
    # manually create a comparison dataframe
    comparison_rows = []
    
    # Get unique masked texts
    masked_texts = all_results_df["masked_text"].unique()
    models = all_results_df["model"].unique()
    
    for text in masked_texts:
        row_data = {"Masked Text": text}
        
        for model in models:
            model_results = all_results_df[(all_results_df["masked_text"] == text) & 
                                           (all_results_df["model"] == model)]
            
            if len(model_results) > 0:
                result = model_results.iloc[0]
                row_data[f"{model} - Top Prediction"] = result["top_1_prediction"]
                row_data[f"{model} - Score"] = result["top_1_score"]
                row_data[f"{model} - Age"] = result["predicted_age"]
                row_data[f"{model} - Region"] = result["predicted_region"]
        
        comparison_rows.append(row_data)
    
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Save to CSV
    summary_df.to_csv(f"{SAVE_FIGS_PATH}prediction_summary.csv", index=False)
    comparison_df.to_csv(f"{SAVE_FIGS_PATH}model_comparison.csv", index=False)
    
    return summary_df, comparison_df

# ===== Generate Model Comparison Visualization =====
def generate_model_comparison_viz(all_results_df):
    """
    Generate visualizations comparing model predictions
    """
    print("Generating model comparison visualizations...")
    
    # Create directory for figures if it doesn't exist
    import os
    os.makedirs(SAVE_FIGS_PATH, exist_ok=True)
    
    # Get unique models
    models = all_results_df["model"].unique()
    
    if len(models) >= 2:  # Only compare if we have at least 2 models
        # Create a visualization showing score comparison
        plt.figure(figsize=(12, 8))
        
        # Group by masked text and model, then get mean score
        score_data = []
        
        for text in all_results_df["masked_text"].unique():
            text_data = {"Masked Text": text}
            
            for model in models:
                model_data = all_results_df[(all_results_df["masked_text"] == text) & 
                                          (all_results_df["model"] == model)]
                
                if len(model_data) > 0:
                    text_data[model] = model_data["top_1_score"].values[0]
            
            score_data.append(text_data)
        
        # Convert to DataFrame
        score_df = pd.DataFrame(score_data)
        
        # Prepare data for plotting
        score_df_melted = pd.melt(score_df, 
                                  id_vars=["Masked Text"], 
                                  value_vars=models, 
                                  var_name="Model", 
                                  value_name="Score")
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x="Masked Text", y="Score", hue="Model", data=score_df_melted)
        plt.title("Prediction Confidence Score Comparison")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"{SAVE_FIGS_PATH}model_score_comparison.png")
        plt.close()
        
        # Generate a heatmap showing age and region classifications
        # First for age classifications
        age_data = []
        
        for text in all_results_df["masked_text"].unique():
            row_data = {"Masked Text": text}
            
            for model in models:
                model_data = all_results_df[(all_results_df["masked_text"] == text) & 
                                          (all_results_df["model"] == model)]
                
                if len(model_data) > 0:
                    row_data[model] = 1 if model_data["predicted_age"].values[0] == "under 30" else 0
            
            age_data.append(row_data)
        
        age_df = pd.DataFrame(age_data)
        age_matrix = age_df.drop(columns=["Masked Text"]).values
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(age_matrix, annot=True, cmap="coolwarm", fmt=".0f",
                   xticklabels=models, yticklabels=age_df["Masked Text"],
                   cbar_kws={'label': 'under 30 (1) vs over 30 (0)'})
        plt.title("Age Classification Comparison (under 30 = 1, over 30 = 0)")
        plt.tight_layout()
        plt.savefig(f"{SAVE_FIGS_PATH}age_classification_comparison.png")
        plt.close()
        
        # Now for region classifications
        region_data = []
        
        for text in all_results_df["masked_text"].unique():
            row_data = {"Masked Text": text}
            
            for model in models:
                model_data = all_results_df[(all_results_df["masked_text"] == text) & 
                                          (all_results_df["model"] == model)]
                
                if len(model_data) > 0:
                    row_data[model] = 1 if model_data["predicted_region"].values[0] == "urban" else 0
            
            region_data.append(row_data)
        
        region_df = pd.DataFrame(region_data)
        region_matrix = region_df.drop(columns=["Masked Text"]).values
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(region_matrix, annot=True, cmap="coolwarm", fmt=".0f",
                   xticklabels=models, yticklabels=region_df["Masked Text"],
                   cbar_kws={'label': 'urban (1) vs rural (0)'})
        plt.title("Region Classification Comparison (urban = 1, rural = 0)")
        plt.tight_layout()
        plt.savefig(f"{SAVE_FIGS_PATH}region_classification_comparison.png")
        plt.close()
    
    return True

# ===== Main Function =====
def main():
    """Main evaluation function"""
    # Load models
    tokenizer, fine_tuned_model, baseline_model, classifier = load_models()
    
    # Get test samples
    test_samples = get_test_samples()
    
    # Prepare ground truth (using fine-tuned model predictions as reference)
    test_samples_with_ground_truth = prepare_ground_truth(fine_tuned_model, classifier, test_samples)
    
    # Evaluate fine-tuned model
    fine_tuned_metrics, fine_tuned_results = evaluate_predictions(
        "Fine-tuned Code-Switch Model", fine_tuned_model, classifier, test_samples_with_ground_truth
    )
    
    # Evaluate baseline model
    baseline_metrics, baseline_results = evaluate_predictions(
        "XLM-RoBERTa Base", baseline_model, classifier, test_samples_with_ground_truth
    )
    
    # Combine results
    all_results = pd.concat([fine_tuned_results, baseline_results])
    all_results.to_csv(SAVE_RESULTS_PATH, index=False)
    
    # Analyze token distributions
    distribution_analysis = analyze_token_distributions(all_results)
    
    # Generate visualizations
    generate_token_distributions_viz(distribution_analysis)
    
    # Generate prediction tables
    summary_df, comparison_df = generate_prediction_tables(all_results)
    
    # Generate model comparison visualizations
    generate_model_comparison_viz(all_results)
    
    # Print individual example results
    print("\n===== Prediction Examples =====")
    for masked_text in test_samples_with_ground_truth[:5]:  # Show first 5 examples
        text = masked_text["masked_text"]
        print(f"\nExample: {text}")
        
        # Fine-tuned model predictions
        ft_row = fine_tuned_results[fine_tuned_results["masked_text"] == text].iloc[0]
        print(f"Fine-tuned model prediction: '{ft_row['top_1_prediction']}' (score: {ft_row['top_1_score']:.4f})")
        print(f"  -> Age: {ft_row['predicted_age']} (confidence: {ft_row['age_confidence']:.2f})")
        print(f"  -> Region: {ft_row['predicted_region']} (confidence: {ft_row['region_confidence']:.2f})")
        
        # Baseline model predictions
        bl_row = baseline_results[baseline_results["masked_text"] == text].iloc[0]
        print(f"Baseline model prediction: '{bl_row['top_1_prediction']}' (score: {bl_row['top_1_score']:.4f})")
        print(f"  -> Age: {bl_row['predicted_age']} (confidence: {bl_row['age_confidence']:.2f})")
        print(f"  -> Region: {bl_row['predicted_region']} (confidence: {bl_row['region_confidence']:.2f})")
    
    # Print comparison in tabular format
    print("\n===== Prediction Summary =====")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comparison_df.head(len(test_samples_with_ground_truth)))
    
    print(f"\nResults saved to {SAVE_RESULTS_PATH} and visualizations to {SAVE_FIGS_PATH}")

if __name__ == "__main__":
    main()