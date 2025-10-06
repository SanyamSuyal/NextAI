import requests
from bs4 import BeautifulSoup
import json
import time
import os
from urllib.parse import urljoin, urlparse
from pathlib import Path
import pandas as pd

class EducationalDataCollector:
    def __init__(self, output_dir="data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def load_sources(self):
        with open('data/sources.json', 'r') as f:
            return json.load(f)
    
    def scrape_text_content(self, url, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                text = soup.get_text(separator='\n', strip=True)
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                return '\n'.join(lines)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to scrape {url}: {str(e)}")
                    return None
                time.sleep(2)
        return None
    
    def collect_educational_content(self):
        print("Collecting educational content...")
        sources = self.load_sources()
        all_data = []
        
        categories = ['educational_resources', 'career_guidance', 'mental_health', 'exam_preparation']
        
        for category in categories:
            if category in sources:
                print(f"\nProcessing {category}...")
                output_file = self.output_dir / f"{category}.txt"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for source in sources[category]:
                        print(f"  - {source.get('name', 'Unknown')}")
                        f.write(f"Source: {source.get('name', 'Unknown')}\n")
                        f.write(f"Topics: {', '.join(source.get('topics', []))}\n")
                        f.write("-" * 80 + "\n\n")
                
                print(f"Saved to {output_file}")
        
        return all_data
    
    def collect_wikipedia_articles(self, topics):
        print("\nCollecting Wikipedia articles...")
        wikipedia_data = []
        
        base_url = "https://en.wikipedia.org/wiki/"
        
        for topic in topics:
            url = base_url + topic.replace(' ', '_')
            print(f"  - {topic}")
            content = self.scrape_text_content(url)
            
            if content:
                wikipedia_data.append({
                    'topic': topic,
                    'source': 'Wikipedia',
                    'content': content
                })
            
            time.sleep(1)
        
        output_file = self.output_dir / "wikipedia.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in wikipedia_data:
                f.write(f"Topic: {article['topic']}\n")
                f.write(f"Source: {article['source']}\n")
                f.write("-" * 80 + "\n")
                f.write(article['content'] + "\n\n")
        
        print(f"Saved {len(wikipedia_data)} Wikipedia articles to {output_file}")
        return wikipedia_data
    
    def generate_synthetic_conversations(self, num_conversations=1000):
        print(f"\nGenerating {num_conversations} synthetic conversations...")
        
        templates = {
            'career_guidance': [
                "I want to pursue {field}. What should I do?",
                "How can I prepare for {exam}?",
                "What are the best colleges for {field}?",
                "Is {career} a good choice?",
                "How do I get into {institution}?"
            ],
            'mental_health': [
                "I'm feeling stressed about exams.",
                "How can I manage academic pressure?",
                "I'm anxious about my future.",
                "How do I deal with failure?",
                "I feel overwhelmed with studies."
            ],
            'tutoring': [
                "Can you explain {topic}?",
                "I'm struggling with {subject}.",
                "How do I solve {problem_type} problems?",
                "What's the best way to study {subject}?",
                "Can you help me understand {concept}?"
            ]
        }
        
        fields = ["Computer Science", "Mechanical Engineering", "Medicine", "Data Science", "AI/ML"]
        exams = ["JEE Advanced", "NEET", "CAT", "GATE", "UPSC"]
        institutions = ["IIT Bombay", "IIT Delhi", "BITS Pilani", "IIM Ahmedabad", "AIIMS"]
        
        conversations = []
        output_file = self.output_dir / "synthetic_conversations.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(num_conversations):
                category = list(templates.keys())[i % len(templates)]
                template = templates[category][i % len(templates[category])]
                
                if '{field}' in template:
                    question = template.format(field=fields[i % len(fields)])
                elif '{exam}' in template:
                    question = template.format(exam=exams[i % len(exams)])
                elif '{institution}' in template:
                    question = template.format(institution=institutions[i % len(institutions)])
                else:
                    question = template
                
                f.write(f"Q: {question}\n")
                f.write(f"Category: {category}\n")
                f.write("-" * 80 + "\n\n")
        
        print(f"Generated {num_conversations} conversations in {output_file}")
        return conversations
    
    def collect_sample_data(self):
        print("=" * 80)
        print("NextAI Data Collection Tool")
        print("=" * 80)
        
        self.collect_educational_content()
        
        wikipedia_topics = [
            "Indian_Institutes_of_Technology",
            "Joint_Entrance_Examination",
            "NEET",
            "Common_Admission_Test",
            "Graduate_Aptitude_Test_in_Engineering",
            "Career_counseling",
            "Educational_psychology",
            "Student_mental_health",
            "Study_skills",
            "Time_management"
        ]
        
        self.collect_wikipedia_articles(wikipedia_topics)
        
        self.generate_synthetic_conversations(5000)
        
        print("\n" + "=" * 80)
        print("Data collection complete!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        
        print("\nNext steps:")
        print("1. Add your own curated data to the data/raw/ directory")
        print("2. Run preprocess.py to clean and prepare the data")
        print("3. Run validate_dataset.py to check data quality")

def main():
    collector = EducationalDataCollector()
    collector.collect_sample_data()
    
    print("\n" + "=" * 80)
    print("DATA COLLECTION GUIDANCE")
    print("=" * 80)
    print("\nTo reach 1M lines of quality data, you should:")
    print("\n1. Educational Content (400K lines):")
    print("   - NCERT textbooks (PDF extraction)")
    print("   - IIT/BITS lecture notes")
    print("   - Open educational resources")
    print("   - Academic papers and research")
    print("\n2. Career Guidance (300K lines):")
    print("   - Career counseling websites")
    print("   - College review platforms")
    print("   - Success stories and interviews")
    print("   - Industry expert blogs")
    print("\n3. Mental Health (150K lines):")
    print("   - Counseling resources from NIMHANS")
    print("   - Student wellness articles")
    print("   - Stress management guides")
    print("   - Psychology resources")
    print("\n4. Exam Preparation (100K lines):")
    print("   - Previous year questions")
    print("   - Study materials and notes")
    print("   - Preparation strategies")
    print("   - Mock test analysis")
    print("\n5. Community Data (50K lines):")
    print("   - Reddit discussions (r/Indian_Academia)")
    print("   - Quora Q&A")
    print("   - Student forums")
    print("   - Social media discussions")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
