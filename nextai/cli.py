import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="NextAI - Educational and Career Guidance AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nextai train --data data/processed/train.txt
  nextai infer --prompt "How do I prepare for JEE?"
  nextai serve --port 8000
  nextai evaluate --model models/nextai
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', default='training/config.yaml')
    train_parser.add_argument('--data', help='Training data path')
    train_parser.add_argument('--output', help='Output directory')
    
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--model', default='models/nextai')
    infer_parser.add_argument('--prompt', required=True)
    
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--model', default='models/nextai')
    serve_parser.add_argument('--port', type=int, default=8000)
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', default='models/nextai')
    eval_parser.add_argument('--test-data', default='data/processed/test.txt')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        from training.train import main as train_main
        sys.argv = ['train', '--config', args.config]
        if args.data:
            sys.argv.extend(['--train_data', args.data])
        if args.output:
            sys.argv.extend(['--output_dir', args.output])
        train_main()
    
    elif args.command == 'infer':
        from deploy.inference import main as infer_main
        sys.argv = ['infer', '--model_path', args.model, '--prompt', args.prompt]
        infer_main()
    
    elif args.command == 'serve':
        from deploy.api_server import main as serve_main
        sys.argv = ['serve', '--model_path', args.model, '--port', str(args.port)]
        serve_main()
    
    elif args.command == 'evaluate':
        from evaluation.evaluate import main as eval_main
        sys.argv = ['evaluate', '--model_path', args.model, '--test_data', args.test_data]
        eval_main()
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
