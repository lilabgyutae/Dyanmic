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
    parser = argparse.ArgumentParser(description='Intent Classification with Dynamic Intent Mapping')
    parser.add_argument('--retriever_model', type=str, default='all-mpnet-base-v2',
                        help='Sentence transformer model for retrieval')
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Model ID for the language model')
    parser.add_argument('--train_file', type=str, default='dataset/Dialoglue/hwu/train_10.csv',
                        help='Path to training file')
    parser.add_argument('--test_file', type=str, default='dataset/Dialoglue/hwu/test_10.csv',
                        help='Path to test file')
    parser.add_argument('--total_examples', type=int, default=20,
                        help='Number of examples to retrieve')
    return parser.parse_args()

def load_model(model_id: str, device):
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def retrieve_similar_examples(query: str, train_df: pd.DataFrame, retriever: SentenceTransformer, total_examples: int) -> List[Tuple[float, str, str]]:
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    train_embeddings = retriever.encode(train_df['sentence'].tolist(), convert_to_tensor=True)
    
    similarities = []
    for idx, (emb, sent, intent) in enumerate(zip(train_embeddings, train_df['sentence'], train_df['label'])):
        similarity = torch.cosine_similarity(query_embedding.unsqueeze(0), emb.unsqueeze(0)).item()
        similarities.append((similarity, sent, intent))
    
    return sorted(similarities, key=lambda x: x[0], reverse=True)[:total_examples]

def get_intent_renaming_prompt(examples: List[Tuple[float, str, str]]) -> str:
    # Get all unique intents and their examples
    intent_groups = defaultdict(set)
    for sim, sent, intent in examples:
        intent_groups[intent].add(sent)
    
    # 예시 구성 with better formatting
    groups_str = "Examples by intent:\n"
    for intent, examples in intent_groups.items():
        groups_str += f"\nIntent: {intent}\n"
        for example in examples:
            groups_str += f"- {example}\n"
    
    mapping_format = "\n".join(f"{intent} -> " for intent in sorted(intent_groups.keys()))
    
    prompt = f'''{groups_str}
Analyze each intent and its examples above, please follow these rules for intent mapping:
1. If the current intent name accurately represents its examples, map it to itself
2. If the intent name needs improvement, create a new descriptive name that better represents the examples
3. For new names:
   - Use lowercase letters only
   - Use underscores between words

Your response should list only the mappings in this format:
INTENT MAPPINGS:
{mapping_format}'''
    return prompt

def extract_mappings_from_response(response: str) -> Dict[str, str]:
    mapping = {}
    
    if "INTENT MAPPINGS:" in response:
        sections = response.split("INTENT MAPPINGS:")[1:]
        
        for section in sections:
            clean_section = section
            if "Final" in clean_section:
                clean_section = clean_section.split("Final")[0]
            if "\nNote:" in clean_section:
                clean_section = clean_section.split("\nNote:")[0]
                
            current_mappings = {}
            lines = [line.strip() for line in clean_section.split('\n') if line.strip()]
            valid_mappings = 0
            
            for line in lines:
                if '->' in line:
                    try:
                        if '(' in line:
                            note_start = line.index('(')
                            mapping_part = line[:note_start]
                        else:
                            mapping_part = line
                            
                        old_intent, new_intent = map(str.strip, mapping_part.split('->'))
                        
                        if old_intent and old_intent not in current_mappings:
                            if new_intent:
                                new_intent = new_intent.lower().replace(' ', '_')
                                current_mappings[old_intent] = new_intent
                                valid_mappings += 1
                    except:
                        continue
            
            if valid_mappings > 0:
                mapping = current_mappings
                break
    
    return mapping

def rename_intents(examples: List[Tuple[float, str, str]], model, tokenizer, device) -> Dict[str, str]:
    unique_intents = set(intent for _, _, intent in examples)
    
    prompt = get_intent_renaming_prompt(examples)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=145,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    intent_mapping = extract_mappings_from_response(response)
    
    final_mapping = {}
    for intent in unique_intents:
        if intent in intent_mapping:
            mapped_intent = intent_mapping[intent]
            if any(c in '()[]{}' for c in mapped_intent) or 'and_so_on' in mapped_intent.lower():
                final_mapping[intent] = intent
            else:
                final_mapping[intent] = mapped_intent
        else:
            final_mapping[intent] = intent
    
    return final_mapping

def prompt_template(query: str, examples: List[Tuple[str, str, float]]) -> str:
    intent_examples_str = "\n".join([f'Text: "{sentence}"\nIntent: {mapped_intent}\n' 
                                   for sentence, mapped_intent, _ in examples])
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an AI assistant specialized in intent classification. Your task is to determine the single most likely intent of a given query based on the examples provided. 
Provide only the name of the most probable intent, without any additional text or explanation. The intent list is given below with example queries for each intent.

[Intent List]
{intent_examples_str}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"{query}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
The top 1 most likely intent is:"""
    return prompt

def extract_intent(response: str) -> str:
    if "The top 1 most likely intent is:" in response:
        predicted = response.split("The top 1 most likely intent is:")[-1].strip()
        predicted = predicted.split('\n')[0].strip()
        if not predicted or predicted.isdigit():
            return None
        return predicted
    return None

def safe_label(label: str) -> str:
    if not isinstance(label, str):
        return ""
    
    processed_label = re.sub(r'[^\w\s-]', '', label).strip().lower()
    return processed_label

def main():
    warnings.filterwarnings("ignore")
    
    # Parse command line arguments
    args = parse_args()
    config = {
        'retriever_model': args.retriever_model,
        'model_id': args.model_id,
        'train_file': args.train_file,
        'test_file': args.test_file,
        'total_examples': args.total_examples
    }
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data and models...")
    train_df = pd.read_csv(config['train_file'])
    test_df = pd.read_csv(config['test_file'])
    
    model, tokenizer = load_model(config['model_id'], device)
    retriever = SentenceTransformer(config['retriever_model']).to(device)
    
    results = []
    correct = 0
    total = len(test_df)
    
    for idx, row in tqdm(enumerate(test_df.iterrows(), 1), total=total, desc="Processing queries"):
        _, row = row
        query = row['sentence']
        true_label = row['label']
        
        examples = retrieve_similar_examples(query, train_df, retriever, config['total_examples'])
        true_label_in_examples = any(intent == true_label for _, _, intent in examples)
        
        intent_mapping = rename_intents(examples, model, tokenizer, device)
        reverse_mapping = {v: k for k, v in intent_mapping.items()}
        
        mapped_examples = [(sent, intent_mapping.get(intent, intent), sim) 
                          for sim, sent, intent in examples]
        
        prompt = prompt_template(query, mapped_examples)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                num_return_sequences=1
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_new_intent = extract_intent(response)
        predicted_original_intent = reverse_mapping.get(predicted_new_intent, predicted_new_intent)
        
        predicted_original_intent = safe_label(predicted_original_intent)
        true_label = safe_label(true_label)
        is_correct = predicted_original_intent == true_label
        if is_correct:
            correct += 1
        
        results.append({
            'sentence': query,
            'true_label': true_label,
            'is_correct': is_correct,
            'num_examples': len(examples),
            'selected_intents': [intent for _, _, intent in examples],
            'is_true_label_in_examples': true_label_in_examples,
            'examples': [sent for _, sent, _ in examples],
            'intent_mappings': intent_mapping,
            'new_mapped_intent': predicted_new_intent,
            'final_predicted_intent': predicted_original_intent
        })
    
    print(f"\nFinal Results:")
    print(f"Total cases tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Final accuracy: {(correct/total):7.1%}")

if __name__ == "__main__":
    main()