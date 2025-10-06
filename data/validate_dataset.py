import os
import json
from collections import Counter

class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = []
        self.stats = {}
    
    def load_dataset(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.dataset = content.split('\n\n')
        
        print(f"Loaded {len(self.dataset)} conversations")
    
    def validate(self):
        print("\n=== Dataset Validation ===\n")
        
        total_lines = 0
        total_words = 0
        qa_pairs = 0
        
        for entry in self.dataset:
            lines = entry.split('\n')
            total_lines += len(lines)
            total_words += len(entry.split())
            
            if 'Question:' in entry and 'Answer:' in entry:
                qa_pairs += 1
        
        self.stats = {
            "total_conversations": len(self.dataset),
            "total_lines": total_lines,
            "total_words": total_words,
            "qa_pairs": qa_pairs,
            "avg_lines_per_conversation": round(total_lines / len(self.dataset), 2),
            "avg_words_per_conversation": round(total_words / len(self.dataset), 2)
        }
        
        print(f"✓ Total Conversations: {self.stats['total_conversations']}")
        print(f"✓ Total Lines: {self.stats['total_lines']}")
        print(f"✓ Total Words: {self.stats['total_words']}")
        print(f"✓ Q&A Pairs: {self.stats['qa_pairs']}")
        print(f"✓ Avg Lines/Conversation: {self.stats['avg_lines_per_conversation']}")
        print(f"✓ Avg Words/Conversation: {self.stats['avg_words_per_conversation']}")
        
        if total_lines >= 600000:
            print(f"\n✅ Dataset meets target requirement of 600k+ lines!")
        else:
            print(f"\n⚠️  Dataset has {total_lines} lines, target is 600k+")
        
        return self.stats
    
    def save_validation_report(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\nValidation report saved to: {output_path}")

if __name__ == "__main__":
    validator = DatasetValidator("/workspaces/NextAI/data/processed/training_data.txt")
    validator.load_dataset()
    validator.validate()
    validator.save_validation_report("/workspaces/NextAI/data/processed/validation_report.json")