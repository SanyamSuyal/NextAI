import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path

class ModelEvaluator:
    def __init__(self, model_path, test_data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from {model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loading test data from {test_data_path}")
        with open(test_data_path, 'r') as f:
            content = f.read()
        
        self.test_examples = [ex.strip() for ex in content.split('<|endoftext|>') if ex.strip()]
        print(f"Loaded {len(self.test_examples)} test examples")
    
    def calculate_perplexity(self):
        print("\nCalculating perplexity...")
        
        total_loss = 0
        total_tokens = 0
        
        for example in tqdm(self.test_examples[:100]):
            inputs = self.tokenizer(example, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return perplexity.item(), avg_loss
    
    def evaluate_generation_quality(self, num_samples=10):
        print("\nEvaluating generation quality...")
        
        test_prompts = [
            "How can I prepare for JEE Advanced?",
            "What are the best colleges for Computer Science?",
            "I'm feeling stressed about exams.",
            "Create a roadmap for becoming a data scientist.",
            "How do I get into IIT Bombay?",
            "What should I study for GATE?",
            "I'm anxious about my career choice.",
            "How can I improve my problem-solving skills?",
            "What are the career options after engineering?",
            "How do I balance academics and mental health?"
        ]
        
        generations = []
        
        for prompt in test_prompts[:num_samples]:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=200,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append({
                'prompt': prompt,
                'response': response
            })
        
        return generations
    
    def calculate_metrics(self):
        print("=" * 80)
        print("NextAI Model Evaluation")
        print("=" * 80)
        
        perplexity, avg_loss = self.calculate_perplexity()
        
        print(f"\nPerplexity: {perplexity:.2f}")
        print(f"Average Loss: {avg_loss:.4f}")
        
        generations = self.evaluate_generation_quality()
        
        print("\n" + "=" * 80)
        print("Sample Generations")
        print("=" * 80)
        
        for i, gen in enumerate(generations, 1):
            print(f"\n{i}. Prompt: {gen['prompt']}")
            print(f"   Response: {gen['response'][:200]}...")
            print("-" * 80)
        
        results = {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'sample_generations': generations,
            'num_test_examples': len(self.test_examples)
        }
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate NextAI model")
    parser.add_argument("--model_path", type=str, default="models/nextai",
                       help="Path to trained model")
    parser.add_argument("--test_data", type=str, default="data/processed/test.txt",
                       help="Path to test data")
    parser.add_argument("--output", type=str, default="evaluation/results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(args.model_path, args.test_data)
    results = evaluator.calculate_metrics()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test examples: {results['num_test_examples']}")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Average Loss: {results['avg_loss']:.4f}")
    
    if results['perplexity'] < 25:
        print("\n✓ Model performance is GOOD (perplexity < 25)")
    elif results['perplexity'] < 40:
        print("\n⚠ Model performance is ACCEPTABLE (perplexity < 40)")
    else:
        print("\n✗ Model may need more training (perplexity >= 40)")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
