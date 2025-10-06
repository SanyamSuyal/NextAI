import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path

class NextAI:
    def __init__(self, model_path="models/nextai", device=None):
        self.model_path = model_path
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading NextAI model from {model_path}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompt,
        max_length=200,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1
    ):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        responses = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return responses[0] if num_return_sequences == 1 else responses
    
    def chat(self, message, **kwargs):
        return self.generate(message, **kwargs)
    
    def career_guidance(self, query, **kwargs):
        prompt = f"Career Guidance: {query}"
        return self.generate(prompt, **kwargs)
    
    def mental_health_support(self, query, **kwargs):
        prompt = f"Mental Health Support: {query}"
        disclaimer = "Note: This is AI-generated advice. Please consult a professional for serious concerns.\n\n"
        response = self.generate(prompt, **kwargs)
        return disclaimer + response
    
    def create_roadmap(self, goal, **kwargs):
        prompt = f"Create a detailed roadmap for: {goal}"
        return self.generate(prompt, max_length=400, **kwargs)
    
    def tutoring_help(self, subject, topic, **kwargs):
        prompt = f"Help with {subject} - {topic}:"
        return self.generate(prompt, **kwargs)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NextAI Inference")
    parser.add_argument("--model_path", type=str, default="models/nextai",
                       help="Path to trained model")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt")
    parser.add_argument("--max_length", type=int, default=200,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    ai = NextAI(model_path=args.model_path, device=args.device)
    
    print(f"\nPrompt: {args.prompt}")
    print("\nResponse:")
    print("-" * 80)
    
    response = ai.generate(
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print(response)
    print("-" * 80)

if __name__ == "__main__":
    main()
