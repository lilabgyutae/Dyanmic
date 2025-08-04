import pandas as pd
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Data preprocessing for Hint3 datasets')
    parser.add_argument('--dataset_path', type=str, default='dataset/Hint3/v3/train',
                        help='Path to the dataset directory')
    parser.add_argument('--file_name', type=str, required=True,
                        help='CSV file name to process (e.g., curekart_train.csv)')
    parser.add_argument('--remove_label', type=str, default='no_nodes_detected',
                        help='Label to remove from dataset')
    return parser.parse_args()

def preprocess_data(file_path, remove_label=None):

    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original labels: {df['label'].value_counts().to_dict()}")
    
    # Convert labels to lowercase
    print("Converting labels to lowercase...")
    df["label"] = df["label"].apply(lambda x: x.lower())
    
    # Remove specified label if provided
    if remove_label:
        original_count = len(df)
        df = df[df['label'] != remove_label.lower()]
        removed_count = original_count - len(df)
        print(f"Removed {removed_count} rows with label '{remove_label}'")
    
    print(f"Processed dataset shape: {df.shape}")
    print(f"Processed labels: {df['label'].value_counts().to_dict()}")
    
    # Save the processed file
    df.to_csv(file_path, index=False)
    print(f"Saved processed data to: {file_path}")
    
    # Display first few rows
    print("\nFirst 5 rows of processed data:")
    print(df.head())
    
    return df

def main():
    args = parse_args()
    
    # Construct full file path
    file_path = os.path.join(args.dataset_path, args.file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist!")
        return
    
    # Process the data
    preprocess_data(
        file_path=file_path,
        remove_label=args.remove_label
    )

if __name__ == "__main__":
    main()