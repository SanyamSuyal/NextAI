import sys
sys.path.append('..')
from deploy.inference import NextAI

def career_guidance_example():
    print("=" * 80)
    print("Career Guidance Example")
    print("=" * 80)
    
    ai = NextAI(model_path="models/nextai")
    
    queries = [
        "I want to get into IIT Bombay Computer Science. What should be my strategy?",
        "What are the career prospects after doing BTech in Mechanical Engineering?",
        "How can I prepare for UPSC while doing my college degree?",
        "Is MBA from IIM worth it? What's the ROI?",
        "I'm confused between medicine and engineering. How do I decide?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 80)
        
        response = ai.career_guidance(query)
        print(f"Response:\n{response}")
        print("=" * 80)

def mental_health_example():
    print("\n" + "=" * 80)
    print("Mental Health Support Example")
    print("=" * 80)
    
    ai = NextAI(model_path="models/nextai")
    
    queries = [
        "I'm feeling very stressed about my upcoming exams.",
        "I failed in my semester exams and feeling demotivated.",
        "How can I deal with peer pressure about career choices?",
        "I'm anxious about my future and can't focus on studies.",
        "How do I maintain work-life balance during exam preparation?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 80)
        
        response = ai.mental_health_support(query)
        print(f"Response:\n{response}")
        print("=" * 80)

def tutoring_example():
    print("\n" + "=" * 80)
    print("Tutoring Help Example")
    print("=" * 80)
    
    ai = NextAI(model_path="models/nextai")
    
    topics = [
        ("Physics", "Newton's Laws of Motion"),
        ("Mathematics", "Calculus and Derivatives"),
        ("Chemistry", "Chemical Bonding"),
        ("Computer Science", "Data Structures and Algorithms"),
        ("English", "Essay Writing Techniques")
    ]
    
    for i, (subject, topic) in enumerate(topics, 1):
        print(f"\n{i}. Subject: {subject}, Topic: {topic}")
        print("-" * 80)
        
        response = ai.tutoring_help(subject, topic)
        print(f"Response:\n{response}")
        print("=" * 80)

def roadmap_example():
    print("\n" + "=" * 80)
    print("Roadmap Generation Example")
    print("=" * 80)
    
    ai = NextAI(model_path="models/nextai")
    
    goals = [
        "becoming a data scientist from scratch",
        "cracking JEE Advanced and getting into IIT",
        "transitioning from engineering to management",
        "preparing for GATE in Computer Science",
        "building a career in artificial intelligence"
    ]
    
    for i, goal in enumerate(goals, 1):
        print(f"\n{i}. Goal: {goal}")
        print("-" * 80)
        
        response = ai.create_roadmap(goal)
        print(f"Response:\n{response}")
        print("=" * 80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NextAI Examples")
    parser.add_argument("--example", type=str, 
                       choices=["career", "mental-health", "tutoring", "roadmap", "all"],
                       default="all",
                       help="Which example to run")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NextAI Usage Examples")
    print("=" * 80)
    print("\nNote: These examples require a trained model at 'models/nextai'")
    print("If you haven't trained the model yet, run training/train.py first")
    print("=" * 80)
    
    if args.example == "all" or args.example == "career":
        career_guidance_example()
    
    if args.example == "all" or args.example == "mental-health":
        mental_health_example()
    
    if args.example == "all" or args.example == "tutoring":
        tutoring_example()
    
    if args.example == "all" or args.example == "roadmap":
        roadmap_example()
    
    print("\n" + "=" * 80)
    print("Examples Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
