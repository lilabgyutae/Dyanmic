import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
import json
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import re
import warnings
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Static Intent Renaming')
    parser.add_argument('--model_id', type=str, default='model',
                        help='Model ID for the language model')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training file')
    parser.add_argument('--mapping_output_file', type=str, required=True,
                        help='Path to save intent mapping CSV file')
    parser.add_argument('--renamed_data_output_file', type=str, required=True,
                        help='Path to save renamed training data CSV file')
    return parser.parse_args()

def load_model(model_id: str, device):
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def get_static_renaming_prompt(intent: str, examples: list) -> str:
    examples_str = "\nExamples for this intent:\n"
    for example in examples:
        examples_str += f"- {example}\n"
    
    # Llama3 형식의 프롬프트
    prompt = f"""<|system|>
Intent name: {intent}
{examples_str}
Analyze the intent name and its examples above, please follow these rules for intent renaming:
1. If the current intent name accurately represents its examples, keep it as is
2. If the intent name needs improvement, create a new descriptive name that better represents the examples
3. For new names:
   - Use lowercase letters only
   - Use underscores between words

<|user|>
Provide only the new intent name in this format:
INTENT MAPPING:
{intent} -> 

<|assistant|>
INTENT MAPPING:"""
    return prompt

def extract_mapping_from_response(response: str, original_intent: str) -> str:
    """Improved response extraction for Llama3"""
    if "INTENT MAPPING:" not in response:
        return original_intent
        
    # Get the last occurrence of INTENT MAPPING: section
    mapping_section = response.split("INTENT MAPPING:")[-1].strip()
    
    # Look for the arrow pattern
    if "->" not in mapping_section:
        return original_intent
        
    try:
        # Split on arrow and get the right side
        mapping = mapping_section.split("->")[1].strip()
        
        # Clean up the mapping
        # Remove any response continuation markers
        mapping = mapping.split('\n')[0].strip()
        mapping = mapping.split('<')[0].strip()  # Remove any XML-like tags
        mapping = mapping.split('(')[0].strip()  # Remove any parenthetical content
        
        # Skip if the mapping is 'assistant' or empty
        if mapping.lower() == 'assistant' or not mapping:
            return original_intent
            
        # Convert to proper format
        cleaned_mapping = mapping.lower().replace(' ', '_')
        # Keep only alphanumeric and underscore
        cleaned_mapping = ''.join(c for c in cleaned_mapping if c.isalnum() or c == '_')
        
        # Additional validation
        if len(cleaned_mapping) > 2 and cleaned_mapping != 'assistant':
            return cleaned_mapping
            
    except Exception as e:
        print(f"Error processing mapping: {e}")
    
    return original_intent

def static_rename_intents(train_df: pd.DataFrame, model, tokenizer, device) -> dict:
    intent_groups = defaultdict(list)
    for _, row in train_df.iterrows():
        intent_groups[row['label']].append(row['sentence'])
    
    intent_mapping = {}
    print("\nProcessing each intent for static renaming...")
    
    for intent in tqdm(intent_groups.keys(), desc="Renaming intents"):
        examples = intent_groups[intent]
        prompt = get_static_renaming_prompt(intent, examples)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_intent = extract_mapping_from_response(response, intent)
        intent_mapping[intent] = new_intent
    
    return intent_mapping

def main():
    warnings.filterwarnings("ignore")
    
    # Parse command line arguments
    args = parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_id, device)
    
    # Read training data
    print(f"\nReading training data from {args.train_file}")
    train_df = pd.read_csv(args.train_file)
    
    # Perform static intent renaming
    intent_mapping = static_rename_intents(train_df, model, tokenizer, device)
    
    # Print mapping results
    print("\nIntent Mapping Results:")
    print("-" * 60)
    for original, renamed in intent_mapping.items():
        print(f"{original:30} -> {renamed}")
    
    # Create mappings DataFrame and save
    mapping_df = pd.DataFrame([
        {'original_intent': k, 'renamed_intent': v}
        for k, v in intent_mapping.items()
    ])
    mapping_df.to_csv(args.mapping_output_file, index=False)
    print(f"\nIntent mappings saved to {args.mapping_output_file}")
    
    # Create renamed training data and save
    train_df_renamed = train_df.copy()
    train_df_renamed['label'] = train_df_renamed['label'].map(intent_mapping)
    train_df_renamed.to_csv(args.renamed_data_output_file, index=False)
    print(f"Renamed training data saved to {args.renamed_data_output_file}")
    
    print(f"\nTotal intents processed: {len(intent_mapping)}")

if __name__ == "__main__":
    main()