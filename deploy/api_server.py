from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
sys.path.append('..')
from deploy.inference import NextAI
import argparse

app = Flask(__name__)
CORS(app)

ai_model = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': ai_model is not None})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        max_length = data.get('max_length', 200)
        temperature = data.get('temperature', 0.7)
        
        response = ai_model.chat(
            message,
            max_length=max_length,
            temperature=temperature
        )
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/career-guidance', methods=['POST'])
def career_guidance():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        response = ai_model.career_guidance(query)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/mental-health', methods=['POST'])
def mental_health():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        response = ai_model.mental_health_support(query)
        
        return jsonify({
            'response': response,
            'status': 'success',
            'disclaimer': 'This is AI-generated advice. Please consult a professional for serious concerns.'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/roadmap', methods=['POST'])
def roadmap():
    try:
        data = request.json
        goal = data.get('goal', '')
        
        if not goal:
            return jsonify({'error': 'No goal provided'}), 400
        
        response = ai_model.create_roadmap(goal)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tutoring', methods=['POST'])
def tutoring():
    try:
        data = request.json
        subject = data.get('subject', '')
        topic = data.get('topic', '')
        
        if not subject or not topic:
            return jsonify({'error': 'Subject and topic required'}), 400
        
        response = ai_model.tutoring_help(subject, topic)
        
        return jsonify({
            'response': response,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description="NextAI API Server")
    parser.add_argument("--model_path", type=str, default="models/nextai",
                       help="Path to trained model")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to run server on")
    
    args = parser.parse_args()
    
    global ai_model
    print("Loading NextAI model...")
    ai_model = NextAI(model_path=args.model_path)
    print("Model loaded successfully!")
    
    print(f"\nStarting API server on {args.host}:{args.port}")
    print(f"\nAvailable endpoints:")
    print(f"  POST /chat - General chat")
    print(f"  POST /career-guidance - Career guidance")
    print(f"  POST /mental-health - Mental health support")
    print(f"  POST /roadmap - Generate roadmaps")
    print(f"  POST /tutoring - Tutoring help")
    print(f"  GET  /health - Health check")
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
