import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import wandb

class NextAITrainer:
    def __init__(self, config_path="training/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_tokenizer_and_model(self):
        print("Loading tokenizer and model...")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.config['model_name'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = GPT2LMHeadModel.from_pretrained(self.config['model_name'])
        
        print(f"Model loaded: {self.config['model_name']}")
        print(f"Model parameters: {self.model.num_parameters() / 1e6:.2f}M")
        
        return self.tokenizer, self.model
    
    def load_and_prepare_dataset(self):
        print("Loading dataset...")
        
        data_files = {
            "train": self.config['data']['train_file'],
            "validation": self.config['data']['val_file']
        }
        
        dataset = load_dataset("text", data_files=data_files)
        
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['validation'])}")
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config['data']['max_length'],
                padding="max_length"
            )
        
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return tokenized_dataset
    
    def setup_training_args(self):
        output_dir = self.config['output_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        training_config = self.config['training']
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config['num_epochs'],
            per_device_train_batch_size=training_config['batch_size'],
            per_device_eval_batch_size=training_config['batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            warmup_steps=training_config['warmup_steps'],
            weight_decay=training_config['weight_decay'],
            max_grad_norm=training_config['max_grad_norm'],
            
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            logging_steps=training_config['logging_steps'],
            save_total_limit=training_config['save_total_limit'],
            
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            
            fp16=self.config['optimization']['fp16'] and torch.cuda.is_available(),
            gradient_checkpointing=self.config['optimization']['gradient_checkpointing'],
            
            report_to="wandb" if 'wandb' in self.config else "none",
            
            logging_dir=f"{output_dir}/logs",
            push_to_hub=False,
        )
        
        return training_args
    
    def train(self):
        print("=" * 80)
        print("NextAI Training Pipeline")
        print("=" * 80)
        
        if 'wandb' in self.config:
            wandb.init(
                project=self.config['wandb']['project'],
                config=self.config
            )
        
        tokenizer, model = self.load_tokenizer_and_model()
        
        tokenized_dataset = self.load_and_prepare_dataset()
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        training_args = self.setup_training_args()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
        )
        
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80)
        
        train_result = trainer.train()
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)
        print(f"Final loss: {train_result.training_loss:.4f}")
        
        final_model_path = self.config['output_dir']
        print(f"\nSaving final model to {final_model_path}...")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        print("\n" + "=" * 80)
        print("Evaluating model...")
        print("=" * 80)
        
        eval_results = trainer.evaluate()
        
        print(f"Evaluation loss: {eval_results['eval_loss']:.4f}")
        
        perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))
        print(f"Perplexity: {perplexity:.2f}")
        
        print("\n" + "=" * 80)
        print("Training Summary")
        print("=" * 80)
        print(f"Model: {self.config['model_name']}")
        print(f"Training examples: {len(tokenized_dataset['train'])}")
        print(f"Validation examples: {len(tokenized_dataset['validation'])}")
        print(f"Epochs: {self.config['training']['num_epochs']}")
        print(f"Final training loss: {train_result.training_loss:.4f}")
        print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(f"Model saved to: {final_model_path}")
        print("=" * 80)
        
        if 'wandb' in self.config:
            wandb.finish()
        
        return trainer

def main():
    parser = argparse.ArgumentParser(description="Train NextAI model")
    parser.add_argument("--config", type=str, default="training/config.yaml",
                       help="Path to config file")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (overrides config)")
    parser.add_argument("--train_data", type=str, default=None,
                       help="Training data path (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (overrides config)")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    args = parser.parse_args()
    
    trainer = NextAITrainer(args.config)
    
    if args.model_name:
        trainer.config['model_name'] = args.model_name
    if args.train_data:
        trainer.config['data']['train_file'] = args.train_data
    if args.output_dir:
        trainer.config['output_dir'] = args.output_dir
    if args.num_epochs:
        trainer.config['training']['num_epochs'] = args.num_epochs
    if args.batch_size:
        trainer.config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        trainer.config['training']['learning_rate'] = args.learning_rate
    
    trainer.train()

if __name__ == "__main__":
    main()
