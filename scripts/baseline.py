import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import json
from tqdm import tqdm
from datetime import datetime
import warnings
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Intent Classification')
    parser.add_argument('--retriever_model', type=str, default='all-mpnet-base-v2',
                        help='Sentence transformer model for retrieval')
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct',
                        help='Model ID for the language model')
    parser.add_argument('--train_file', type=str, required=True,
                        help='Path to training file')
    parser.add_argument('--test_file', type=str, required=True,
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

def prompt_template(query: str, examples: List[Tuple[str, str, float]]) -> str:
    intent_examples_str = "\n".join([f'Text: "{sentence}"\nIntent: {intent}\n' 
                                   for _, sentence, intent in examples])
    
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

def extract_intent(model_output: str) -> str:
    return model_output.split("The top 1 most likely intent is:")[-1].strip()

def retrieve_examples(query: str, true_label: str, train_df: pd.DataFrame, 
                     retriever: SentenceTransformer, total_examples: int) -> Tuple[List[Tuple[str, str, float]], List[str], bool]:
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    train_embeddings = retriever.encode(train_df['sentence'].tolist(), convert_to_tensor=True)
    
    similarities = []
    for idx, (emb, sent, intent) in enumerate(zip(train_embeddings, train_df['sentence'], train_df['label'])):
        similarity = torch.cosine_similarity(query_embedding.unsqueeze(0), emb.unsqueeze(0)).item()
        similarities.append((similarity, sent, intent))
    
    top_k = sorted(similarities, key=lambda x: x[0], reverse=True)[:total_examples]
    sorted_examples = sorted(top_k, key=lambda x: x[0])
    
    selected_intents = list(set([intent.lower() for _, _, intent in sorted_examples]))
    is_true_label_in_examples = true_label.lower() in selected_intents
    
    return sorted_examples, selected_intents, is_true_label_in_examples

def predict_intent(query: str, examples: List[Tuple[str, str, float]], model, tokenizer, device) -> Tuple[str, str]:
    prompt = prompt_template(query, examples)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            num_return_sequences=1,
            output_scores=False,
            return_dict_in_generate=True
        )
    
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    predicted_intent = extract_intent(response)
    
    return predicted_intent, prompt

def safe_lower(x):
    if isinstance(x, str):
        return x.lower()
    elif pd.isna(x):
        return ''
    else:
        return str(x).lower()

def main():
    warnings.filterwarnings("ignore")
    
    # Parse command line arguments
    args = parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading data...")
    train_df = pd.read_csv(args.train_file)
    test_df = pd.read_csv(args.test_file)
    
    print("Setting up model and retriever...")
    retriever = SentenceTransformer(args.retriever_model).to(device)
    model, tokenizer = load_model(args.model_id, device)
    model = model.to(device)
    
    results = []
    correct = 0
    total = len(test_df)
    
    for idx, row in tqdm(test_df.iterrows(), total=total, desc="Processing queries"):
        query = row['sentence']
        true_label = safe_lower(row['label'])
        
        examples, selected_intents, is_true_label_in_examples = retrieve_examples(
            query, true_label, train_df, retriever, args.total_examples)
        predicted_intent, full_prompt = predict_intent(query, examples, model, tokenizer, device)
        predicted_intent = safe_lower(predicted_intent)
        
        is_correct = predicted_intent == true_label
        if is_correct:
            correct += 1
        
        result = {
            'sentence': query,
            'true_label': true_label,
            'predicted_intent': predicted_intent,
            'is_correct': is_correct,
            'num_examples': len(examples),
            'selected_intents': ", ".join(selected_intents),
            'is_true_label_in_examples': is_true_label_in_examples,
            'examples': str([(ex[1], ex[2]) for ex in examples]),
            'examples_with_prob': "; ".join([f"({ex[1]},{ex[2]},{ex[0]:.4f})" for ex in examples])
        }
        results.append(result)
    
    final_accuracy = (correct / total) * 100
    
    print(f"\nFinal Results:")
    print(f"Total cases tested: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Final accuracy: {final_accuracy:.2f}%")
    print(f"Average number of examples used: {sum(r['num_examples'] for r in results) / total:.2f}")

if __name__ == "__main__":
    main()