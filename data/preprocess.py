import os
import re
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

class DataPreprocessor:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s\.\,\?\!\-\:\;\'\"]', '', text)
        text = text.strip()
        return text
    
    def filter_quality(self, text, min_length=50, max_length=1000):
        if len(text) < min_length or len(text) > max_length:
            return False
        
        words = text.split()
        if len(words) < 10:
            return False
        
        if text.count('.') < 1 and text.count('?') < 1:
            return False
        
        return True
    
    def create_training_format(self, texts):
        formatted_data = []
        
        for text in texts:
            if self.filter_quality(text):
                formatted_data.append(text + "\n<|endoftext|>\n")
        
        return formatted_data
    
    def process_file(self, input_file):
        print(f"Processing {input_file}...")
        
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        sections = content.split('-' * 80)
        
        cleaned_texts = []
        for section in tqdm(sections, desc="Cleaning"):
            cleaned = self.clean_text(section)
            if cleaned:
                cleaned_texts.append(cleaned)
        
        return cleaned_texts
    
    def process_all_files(self):
        print("=" * 80)
        print("NextAI Data Preprocessing")
        print("=" * 80)
        
        all_texts = []
        
        raw_files = list(self.raw_dir.glob("*.txt"))
        
        if not raw_files:
            print(f"\nNo files found in {self.raw_dir}")
            print("Please run collect_educational_data.py first")
            return
        
        for file_path in raw_files:
            texts = self.process_file(file_path)
            all_texts.extend(texts)
        
        print(f"\nTotal texts collected: {len(all_texts)}")
        
        formatted_data = self.create_training_format(all_texts)
        print(f"Texts after quality filtering: {len(formatted_data)}")
        
        train_split = int(0.9 * len(formatted_data))
        val_split = int(0.95 * len(formatted_data))
        
        train_data = formatted_data[:train_split]
        val_data = formatted_data[train_split:val_split]
        test_data = formatted_data[val_split:]
        
        train_file = self.processed_dir / "train.txt"
        val_file = self.processed_dir / "val.txt"
        test_file = self.processed_dir / "test.txt"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            f.writelines(train_data)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            f.writelines(val_data)
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.writelines(test_data)
        
        print("\n" + "=" * 80)
        print("Preprocessing complete!")
        print("=" * 80)
        print(f"\nTrain set: {len(train_data)} examples -> {train_file}")
        print(f"Validation set: {len(val_data)} examples -> {val_file}")
        print(f"Test set: {len(test_data)} examples -> {test_file}")
        
        stats = {
            'total_examples': len(formatted_data),
            'train_examples': len(train_data),
            'val_examples': len(val_data),
            'test_examples': len(test_data),
            'train_lines': sum(text.count('\n') for text in train_data),
            'avg_length': sum(len(text) for text in formatted_data) / len(formatted_data)
        }
        
        stats_file = self.processed_dir / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDataset statistics saved to {stats_file}")
        print(f"\nTotal training lines: {stats['train_lines']:,}")
        print(f"Average text length: {stats['avg_length']:.0f} characters")
        
        return stats

def main():
    preprocessor = DataPreprocessor()
    stats = preprocessor.process_all_files()
    
    if stats:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Run validate_dataset.py to check data quality")
        print("2. Add more data to reach 1M lines target")
        print("3. Run training/train.py or use training/train_colab.ipynb")
        print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
