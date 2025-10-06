import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

class DatasetValidator:
    def __init__(self, processed_dir="data/processed"):
        self.processed_dir = Path(processed_dir)
        
    def load_dataset(self, split="train"):
        file_path = self.processed_dir / f"{split}.txt"
        
        if not file_path.exists():
            print(f"Error: {file_path} not found")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        examples = content.split('<|endoftext|>')
        examples = [ex.strip() for ex in examples if ex.strip()]
        
        return examples
    
    def analyze_dataset(self, examples):
        stats = {
            'total_examples': len(examples),
            'total_characters': sum(len(ex) for ex in examples),
            'total_words': sum(len(ex.split()) for ex in examples),
            'total_lines': sum(ex.count('\n') for ex in examples),
            'avg_length': np.mean([len(ex) for ex in examples]),
            'median_length': np.median([len(ex) for ex in examples]),
            'min_length': min(len(ex) for ex in examples),
            'max_length': max(len(ex) for ex in examples),
            'avg_words': np.mean([len(ex.split()) for ex in examples]),
        }
        
        return stats
    
    def check_data_quality(self, examples):
        issues = []
        
        if len(examples) == 0:
            issues.append("Dataset is empty")
            return issues
        
        short_examples = sum(1 for ex in examples if len(ex) < 50)
        if short_examples > len(examples) * 0.1:
            issues.append(f"Warning: {short_examples} examples are very short (< 50 chars)")
        
        long_examples = sum(1 for ex in examples if len(ex) > 2000)
        if long_examples > len(examples) * 0.05:
            issues.append(f"Warning: {long_examples} examples are very long (> 2000 chars)")
        
        duplicates = len(examples) - len(set(examples))
        if duplicates > 0:
            issues.append(f"Warning: {duplicates} duplicate examples found")
        
        non_english = sum(1 for ex in examples if not self.is_mostly_english(ex))
        if non_english > len(examples) * 0.2:
            issues.append(f"Info: {non_english} examples contain non-English text")
        
        return issues if issues else ["All quality checks passed!"]
    
    def is_mostly_english(self, text):
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.7 if text else True
    
    def validate_all_splits(self):
        print("=" * 80)
        print("NextAI Dataset Validation")
        print("=" * 80)
        
        for split in ['train', 'val', 'test']:
            print(f"\n{'='*80}")
            print(f"Validating {split.upper()} set")
            print('='*80)
            
            examples = self.load_dataset(split)
            
            if examples is None:
                continue
            
            stats = self.analyze_dataset(examples)
            
            print(f"\nDataset Statistics:")
            print(f"  Total examples: {stats['total_examples']:,}")
            print(f"  Total characters: {stats['total_characters']:,}")
            print(f"  Total words: {stats['total_words']:,}")
            print(f"  Total lines: {stats['total_lines']:,}")
            print(f"  Average length: {stats['avg_length']:.0f} characters")
            print(f"  Median length: {stats['median_length']:.0f} characters")
            print(f"  Min length: {stats['min_length']} characters")
            print(f"  Max length: {stats['max_length']} characters")
            print(f"  Average words per example: {stats['avg_words']:.0f}")
            
            issues = self.check_data_quality(examples)
            
            print(f"\nQuality Checks:")
            for issue in issues:
                print(f"  {issue}")
        
        print("\n" + "=" * 80)
        print("DATASET SIZE ESTIMATION")
        print("=" * 80)
        
        train_examples = self.load_dataset('train')
        if train_examples:
            stats = self.analyze_dataset(train_examples)
            current_lines = stats['total_lines']
            target_lines = 1_000_000
            
            print(f"\nCurrent training lines: {current_lines:,}")
            print(f"Target: {target_lines:,} lines")
            print(f"Progress: {(current_lines / target_lines * 100):.1f}%")
            print(f"Remaining: {target_lines - current_lines:,} lines needed")
            
            if current_lines < target_lines:
                multiplier = target_lines / current_lines
                print(f"\nYou need approximately {multiplier:.1f}x more data")
                print("\nSuggestions to reach 1M lines:")
                print("  1. Scrape more educational websites")
                print("  2. Add PDF textbook content")
                print("  3. Include more community discussions")
                print("  4. Generate synthetic Q&A pairs")
                print("  5. Add multilingual content")
        
        print("\n" + "=" * 80)

def main():
    validator = DatasetValidator()
    validator.validate_all_splits()
    
    print("\n" + "=" * 80)
    print("TRAINING LOSS ESTIMATION")
    print("=" * 80)
    print("\nFor GPT-2 Medium (355M params) on 1M lines:")
    print("\nExpected Training Loss:")
    print("  - Initial loss: ~3.5-4.0")
    print("  - After 1 epoch: ~2.0-2.5")
    print("  - After 2 epochs: ~1.5-2.0")
    print("  - After 3 epochs: ~1.2-1.8 (convergence)")
    print("\nFactors affecting loss:")
    print("  - Data quality and diversity")
    print("  - Domain-specific vocabulary")
    print("  - Training hyperparameters")
    print("  - Model architecture")
    print("\nPerplexity estimation:")
    print("  - Target perplexity: < 25")
    print("  - Good performance: 15-20")
    print("  - Excellent performance: < 15")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
