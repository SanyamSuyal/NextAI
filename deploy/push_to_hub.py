from huggingface_hub import HfApi, login, create_repo
from pathlib import Path
import argparse
import shutil
import json

def push_to_hugging_face(model_path, repo_name, token=None, private=False):
    print("=" * 80)
    print("Pushing NextAI to Hugging Face Hub")
    print("=" * 80)
    
    if token:
        login(token=token)
    else:
        print("\nPlease login to Hugging Face:")
        login()
    
    print(f"\nModel path: {model_path}")
    print(f"Repository: {repo_name}")
    
    try:
        create_repo(repo_name, private=private, exist_ok=True)
        print(f"Repository created/verified: {repo_name}")
    except Exception as e:
        print(f"Repository may already exist: {e}")
    
    api = HfApi()
    
    model_path = Path(model_path)
    
    files_to_upload = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    ]
    
    print("\nUploading files to Hugging Face Hub...")
    
    for file_name in files_to_upload:
        file_path = model_path / file_name
        if file_path.exists():
            print(f"  Uploading {file_name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name,
                repo_id=repo_name,
                repo_type="model"
            )
        else:
            print(f"  Warning: {file_name} not found")
    
    readme_content = f"""---
language: en
license: mit
tags:
- education
- career-guidance
- mental-health
- tutoring
- gpt2
datasets:
- custom
---

# NextAI

NextAI is an intelligent conversational AI model designed for educational and career guidance, specifically tailored for the Indian education ecosystem.

## Model Description

- **Base Model**: GPT-2 Medium (355M parameters)
- **Fine-tuned on**: 1M+ lines of educational content, career guidance, and mental health resources
- **Primary Use Cases**: Career counseling, exam preparation, tutoring, mental wellness support

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "{repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "How can I prepare for JEE Advanced?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

## Training Data

The model was trained on diverse educational content including:
- Career guidance articles
- Educational resources (NCERT, IIT, BITS syllabi)
- Mental health and wellness resources
- Exam preparation materials (JEE, NEET, GATE, CAT)
- Student counseling conversations

## Limitations

- Primarily trained on Indian education system data
- Should not replace professional counseling or medical advice
- May occasionally generate inaccurate information
- Best suited for guidance and information, not critical decisions

## Citation

```bibtex
@software{{nextai2025,
  title={{NextAI: An AI Model for Educational and Career Guidance}},
  author={{Suyal, Sanyam}},
  year={{2025}},
  url={{https://github.com/SanyamSuyal/NextAI}}
}}
```

## Links

- **GitHub**: [SanyamSuyal/NextAI](https://github.com/SanyamSuyal/NextAI)
- **Website**: [NextBench](https://nextbench.com)

## License

MIT License
"""
    
    readme_path = model_path / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"  Uploading README.md...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model"
    )
    
    print("\n" + "=" * 80)
    print("Upload complete!")
    print("=" * 80)
    print(f"\nYour model is now available at:")
    print(f"https://huggingface.co/{repo_name}")
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Push NextAI to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--repo_name", type=str, required=True,
                       help="Repository name (username/model-name)")
    parser.add_argument("--token", type=str, default=None,
                       help="Hugging Face API token")
    parser.add_argument("--private", action="store_true",
                       help="Make repository private")
    
    args = parser.parse_args()
    
    push_to_hugging_face(
        model_path=args.model_path,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private
    )

if __name__ == "__main__":
    main()
