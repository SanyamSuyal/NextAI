import sys
import os
sys.path.append('/workspaces/NextAI')

from data.generate_dataset import DatasetGenerator
from data.validate_dataset import DatasetValidator

def main():
    print("=" * 60)
    print("NextAI Dataset Generation & Validation Pipeline")
    print("=" * 60)
    
    print("\n📝 Step 1: Generating dataset...")
    generator = DatasetGenerator()
    dataset = generator.generate_dataset()
    
    output_path = "/workspaces/NextAI/data/processed/training_data.txt"
    generator.save_dataset(output_path)
    
    print("\n✓ Step 2: Validating dataset...")
    validator = DatasetValidator(output_path)
    validator.load_dataset()
    stats = validator.validate()
    validator.save_validation_report("/workspaces/NextAI/data/processed/validation_report.json")
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print(f"\n📁 Dataset Location: {output_path}")
    print(f"📊 Total Lines: {stats['total_lines']:,}")
    print(f"💬 Total Conversations: {stats['total_conversations']:,}")
    print(f"📈 Ready for training!")

if __name__ == "__main__":
    main()