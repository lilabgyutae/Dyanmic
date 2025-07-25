import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import ast
from tqdm import tqdm
from datetime import datetime
import os
import warnings
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Embedding Analysis for Intent Classification Results')
    parser.add_argument('--model_id', type=str, default='model',
                        help='Model ID for embedding analysis')
    parser.add_argument('--dataset_files', type=str, nargs='+', required=True,
                        help='List of CSV files containing intent classification results')
    parser.add_argument('--output_dir', type=str, default='embedding_analysis_results',
                        help='Output directory for results')
    return parser.parse_args()

def pairwise_cos_sim(A, B):
    """Calculate pairwise cosine similarity between two sets of embeddings"""
    dot_product = torch.matmul(A, B.T)
    norm_A = torch.norm(A, dim=1, keepdim=True)
    norm_B = torch.norm(B, dim=1, keepdim=True)
    denom = norm_A * norm_B.T
    cos_sim_matrix = dot_product / (denom + 1e-8)
    return cos_sim_matrix

def load_model(model_id: str, device):
    """Load model and tokenizer"""
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    return model, tokenizer

def process_dataset(model, tokenizer, dataset_path, device):
    """Process a single dataset and calculate embedding similarities"""
    print(f"Processing: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path)
    sim_ori, sim_new = 0, 0
    
    for num in tqdm(range(len(df)), desc=f"Processing {os.path.basename(dataset_path)}"):
        try:
            mapping = ast.literal_eval(df['intent_mappings'][num])
            exs = ast.literal_eval(df['examples'][num])
            intents = ast.literal_eval(df['selected_intents'][num])

            new_intents = [mapping[new_int] for new_int in intents]

            embeddings_original = []
            embeddings_new = []
            
            for (example, intent, new_intent) in zip(exs, intents, new_intents):
                # Original intent embedding
                input_orig = tokenizer([f"Text: {example} \\nIntent: {intent}"], 
                                     return_tensors='pt').to(device)
                # New intent embedding
                input_new = tokenizer([f"Text: {example} \\nIntent: {new_intent}"], 
                                    return_tensors='pt').to(device)
                
                with torch.no_grad():    
                    output_orig = model(**input_orig, output_hidden_states=True).hidden_states[-1][-1]
                    output_new = model(**input_new, output_hidden_states=True).hidden_states[-1][-1]
                
                embeddings_original.append(output_orig)
                embeddings_new.append(output_new)

            # Stack embeddings
            s_1 = torch.stack(embeddings_original)
            s_2 = torch.stack(embeddings_new)

            # Calculate similarities
            sim = pairwise_cos_sim(s_1, s_1)
            sim2 = pairwise_cos_sim(s_2, s_2)

            # Sum upper triangular parts (excluding diagonal)
            num_pairs = len(embeddings_original) * (len(embeddings_original) - 1) // 2
            if num_pairs > 0:
                sim_ori += (sim.triu(diagonal=1).sum() / num_pairs).item()
                sim_new += (sim2.triu(diagonal=1).sum() / num_pairs).item()
        
        except Exception as e:
            print(f"Error processing row {num}: {str(e)}")
            continue
        
    return sim_ori / len(df), sim_new / len(df)

def main():
    warnings.filterwarnings("ignore")
    
    # Parse command line arguments
    args = parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create results directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.output_dir}_{timestamp}"
    individual_results_dir = os.path.join(results_dir, "individual_results")
    os.makedirs(individual_results_dir, exist_ok=True)

    # Initialize model
    model, tokenizer = load_model(args.model_id, device)

    # Process each dataset and store results
    results = []

    for dataset_path in args.dataset_files:
        dataset_name = os.path.basename(dataset_path).replace("_intent_classification_results.csv", "").replace(".csv", "")
        print(f"\\nProcessing dataset: {dataset_name}")
        
        try:
            sim_ori, sim_new = process_dataset(model, tokenizer, dataset_path, device)
            
            # Create individual result for this dataset
            dataset_result = {
                'dataset': dataset_name,
                'original_intent_similarity': sim_ori,
                'new_intent_similarity': sim_new,
                'similarity_difference': sim_new - sim_ori
            }
            results.append(dataset_result)
            
            # Save individual result
            individual_df = pd.DataFrame([dataset_result])
            individual_output = os.path.join(individual_results_dir, f"{dataset_name}_analysis.csv")
            individual_df.to_csv(individual_output, index=False)
            
            print(f"Results for {dataset_name}:")
            print(f"Original intent similarity: {sim_ori:.4f}")
            print(f"New intent similarity: {sim_new:.4f}")
            print(f"Similarity difference: {sim_new - sim_ori:.4f}")
            print(f"Individual results saved to: {individual_output}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")
            continue

    # Save final combined results
    if results:
        results_df = pd.DataFrame(results)
        output_file = os.path.join(results_dir, f"embedding_analysis_combined.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\\nCombined results saved to: {output_file}")

        # Display final results table
        print("\\nFinal Results Summary:")
        print(results_df.to_string(index=False))
        
        # Calculate and display average results
        avg_orig = results_df['original_intent_similarity'].mean()
        avg_new = results_df['new_intent_similarity'].mean()
        avg_diff = results_df['similarity_difference'].mean()
        
        print(f"\\nAverage Results:")
        print(f"Average original intent similarity: {avg_orig:.4f}")
        print(f"Average new intent similarity: {avg_new:.4f}")
        print(f"Average similarity difference: {avg_diff:.4f}")
        
    else:
        print("\\nNo results were generated due to errors.")

if __name__ == "__main__":
    main()