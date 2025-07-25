# Dynamic Intent Classification

This project provides code for comparing Intent Classification performance across various methods including baseline, static, and dynamic approaches.

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Structure

```
dynamic/
├── dataset/
│   ├── Dialoglue/
│   │   ├── hwu/
│   │   │   ├── train_10.csv
│   │   │   └── test.csv
│   │   │  
│   │   ├── banking/
│   │   ├── clinc/
│   │   └── ...
│   └── Hint3/
│           └── train/
│               ├── curekart_train.csv
│               ├── powerplay11_train.csv
│               └── sofmattress_train.csv
└── scripts/
    ├── baseline.py
    ├── model.py (dynamic)
    ├── static_naming.py
    ├── static_model.py
    └── ...
```

## Scripts Description and Usage

## Scripts and Usage

### Data Preprocessing(for Hint3)
```bash
python datapreprocessing.py --file_name curekart_train.csv
```

### Baseline Methods
```bash
python baseline_model.py --train_file dataset/Dialoglue/hwu/train_10.csv --test_file dataset/Dialoglue/hwu/test.csv --model_id meta-llama/Meta-Llama-3-8B-Instruct

### Static Methods
```bash
# Step 1: Generate intent mappings
python static_naming.py --train_file dataset/Dialoglue/hwu/train_10.csv --mapping_output_file hwu_mapping.csv --renamed_data_output_file hwu_renamed.csv --model_id meta-llama/Meta-Llama-3-8B-Instruct

# Step 2: Classification with pre-renamed data
python static_model.py --train_file hwu_renamed.csv --test_file dataset/Dialoglue/hwu/test.csv --mapping_file hwu_mapping.csv --model_id meta-llama/Meta-Llama-3-8B-Instruct
```

### Dynamic Methods
```bash
# Single model dynamic
python model.py --train_file dataset/Dialoglue/hwu/train_10.csv --test_file dataset/Dialoglue/hwu/test.csv --model_id meta-llama/Meta-Llama-3-8B-Instruct

# Two models: Qwen mapping + Llama classification
python dynamic_two_model.py --train_file dataset/Dialoglue/hwu/train_10.csv --test_file dataset/Dialoglue/hwu/test.csv --mapping_model_id Qwen/Qwen2.5-1.5B-Instruct --classification_model_id meta-llama/Meta-Llama-3-8B-Instruct

# Two models: Llama mapping + Qwen classification
python dynamic_llama_qwen.py --train_file dataset/Dialoglue/hwu/train_10.csv --test_file dataset/Dialoglue/hwu/test.csv --mapping_model_id meta-llama/Meta-Llama-3-8B-Instruct --classification_model_id Qwen/Qwen2.5-7B-Instruct
```

### Embedding Analysis
```bash
python embedding_analysis.py --model_id Qwen/Qwen2.5-7B-Instruct --dataset_files results/*.csv --output_dir embedding_analysis
```

## Key Arguments

**Common Arguments:**
- `--retriever_model`: Sentence transformer model (default: `all-mpnet-base-v2`)
- `--model_id`: Language model ID (default: `model`)
- `--train_file`: Training file path (required)
- `--test_file`: Test file path (required)  
- `--total_examples`: Number of examples to retrieve (default: 20)

**Static Method Arguments:**
- `--mapping_output_file`: File to save intent mappings (required)
- `--renamed_data_output_file`: File to save renamed data (required)
- `--mapping_file`: Intent mapping file for classification (required)

**Dynamic Two-Model Arguments:**
- `--mapping_model_id`: Model for intent renaming (required)
- `--classification_model_id`: Model for final classification (required)

**Analysis Arguments:**
- `--dataset_files`: Result files to analyze (required, multiple allowed)
- `--output_dir`: Output directory (default: `embedding_analysis_results`)

## Execution Order

1. **Preprocessing** → 2. **Baseline** → 3. **Static** → 4. **Dynamic** → 5. **Analysis**

## Results

Each script outputs the following information:
- Total number of test cases
- Number of correct predictions
- Final accuracy (%)
- Average number of examples used

## Notes

- Ensure sufficient GPU memory is available
- Model loading may take some time
- Verify that dataset file paths are correct
