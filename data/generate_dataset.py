import json
import random
from typing import List, Dict
import os
from datetime import datetime

class DatasetGenerator:
    def __init__(self):
        self.dataset = []
        self.line_count = 0
        self.target_lines = 650000
        
        self.indian_schools = [
            "Delhi Public School", "Kendriya Vidyalaya", "Navodaya Vidyalaya",
            "DAV Public School", "Ryan International", "St. Xavier's",
            "Bishop Cotton School", "La Martiniere", "Modern School",
            "Sardar Patel Vidyalaya", "Amity International", "Bal Bharati"
        ]
        
        self.indian_colleges = [
            "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
            "BITS Pilani", "NIT Trichy", "NIT Warangal", "IIIT Hyderabad",
            "IIM Ahmedabad", "IIM Bangalore", "IIM Calcutta", "AIIMS Delhi",
            "JIPMER Puducherry", "Delhi University", "Mumbai University",
            "Jadavpur University", "Anna University", "BHU Varanasi"
        ]
        
        self.subjects = {
            "school": ["Mathematics", "Physics", "Chemistry", "Biology", "English", 
                      "Hindi", "Social Science", "Computer Science", "Economics"],
            "engineering": ["Data Structures", "Algorithms", "Operating Systems", 
                          "DBMS", "Computer Networks", "Machine Learning", "AI",
                          "Thermodynamics", "Mechanics", "Electronics"],
            "medical": ["Anatomy", "Physiology", "Biochemistry", "Pharmacology",
                       "Pathology", "Microbiology", "Surgery", "Medicine"],
            "management": ["Marketing", "Finance", "HR", "Operations", "Strategy",
                         "Business Analytics", "Entrepreneurship"]
        }
        
        self.exams = {
            "JEE Main": {"subjects": ["Physics", "Chemistry", "Mathematics"], "difficulty": "high"},
            "JEE Advanced": {"subjects": ["Physics", "Chemistry", "Mathematics"], "difficulty": "very high"},
            "NEET": {"subjects": ["Physics", "Chemistry", "Biology"], "difficulty": "high"},
            "GATE": {"subjects": ["Engineering subjects", "Aptitude"], "difficulty": "high"},
            "CAT": {"subjects": ["Quantitative", "Verbal", "DILR"], "difficulty": "high"},
            "UPSC": {"subjects": ["GS", "Optional", "Essay"], "difficulty": "very high"},
            "CLAT": {"subjects": ["Legal", "Reasoning", "English"], "difficulty": "medium"},
            "Class 10 Board": {"subjects": ["All subjects"], "difficulty": "medium"},
            "Class 12 Board": {"subjects": ["Stream subjects"], "difficulty": "medium"}
        }

    def generate_educational_content(self, target_lines: int):
        print(f"Generating educational content: {target_lines} lines...")
        categories = [
            self.generate_school_questions,
            self.generate_tuition_queries,
            self.generate_college_queries,
            self.generate_subject_explanations,
            self.generate_homework_help,
            self.generate_concept_clarifications
        ]
        
        lines_per_category = target_lines // len(categories)
        for category_func in categories:
            category_func(lines_per_category)

    def generate_school_questions(self, count: int):
        templates = [
            "Question: How do I solve {} problems in {}?\nAnswer: To solve {} problems in {}, start by understanding the fundamental concepts. Break down complex problems into smaller steps. Practice regularly with NCERT textbook exercises and reference books like RD Sharma or RS Aggarwal. Focus on understanding rather than memorization. Work through solved examples first, then attempt unsolved problems. Create a formula sheet for quick revision. Join study groups or tuition classes if needed for additional support.\n",
            
            "Question: What is the best way to prepare for {} exam in {}?\nAnswer: For {} exam preparation in {}, create a structured study plan covering the entire syllabus. Allocate more time to difficult topics. Use NCERT textbooks as your primary resource. Practice previous year question papers to understand exam patterns. Take regular mock tests to assess your preparation level. Focus on weak areas and revise strong topics periodically. Maintain consistency in your study routine.\n",
            
            "Question: I'm weak in {}. How can I improve?\nAnswer: To improve in {}, start with basics and build a strong foundation. Dedicate 1-2 hours daily specifically for this subject. Use multiple resources like textbooks, online videos, and coaching materials. Practice numerical problems daily if it's a calculation-based subject. Make concise notes for theory subjects. Seek help from teachers or tutors when stuck. Regular revision is key to improvement.\n",
            
            "Question: Which school is better for {} education in India?\nAnswer: For {} education, schools like {}, {}, and {} are highly regarded. Look for schools with experienced faculty, good infrastructure, and strong academic records. Consider factors like student-teacher ratio, extracurricular activities, and board affiliation. Visit schools during admission period to get firsthand experience. Talk to current students and parents. Choose based on your child's needs and learning style rather than just reputation.\n",
            
            "Question: How important is {} in Class {}?\nAnswer: {} in Class {} is crucial as it forms the foundation for higher studies. This subject helps develop analytical thinking and problem-solving skills. It's important for competitive exams like JEE, NEET, or board examinations. Strong grasp of concepts now will make advanced topics easier later. Regular practice and conceptual clarity are more important than rote learning. Don't ignore this subject even if it seems difficult initially.\n"
        ]
        
        for _ in range(count):
            template = random.choice(templates)
            if template.count('{}') == 2:
                subject = random.choice(self.subjects["school"])
                context = random.choice(["class 10", "class 12", "board exams", "school"])
                self.dataset.append(template.format(subject, context, subject, context))
            elif template.count('{}') == 1:
                subject = random.choice(self.subjects["school"])
                self.dataset.append(template.format(subject, subject))
            elif template.count('{}') == 4:
                subject = random.choice(self.subjects["school"])
                school1, school2, school3 = random.sample(self.indian_schools, 3)
                self.dataset.append(template.format(subject, subject, school1, school2, school3))
            elif template.count('{}') == 3:
                subject = random.choice(self.subjects["school"])
                grade = random.choice(["9", "10", "11", "12"])
                self.dataset.append(template.format(subject, grade, subject, grade))
            self.line_count += template.count('\n')

    def generate_tuition_queries(self, count: int):
        templates = [
            "Question: Should I join tuition classes for {}?\nAnswer: Joining tuition for {} depends on your current understanding and school teaching quality. Tuition classes provide structured learning, doubt clearing, and regular practice. They're beneficial if you struggle with self-study or need extra guidance. However, self-study with good resources can be equally effective if you're disciplined. Consider online platforms like Unacademy, Vedantu, or Khan Academy as alternatives. Evaluate your learning style and budget before deciding.\n",
            
            "Question: What are the best tuition centers for {} preparation in India?\nAnswer: Top tuition centers for {} include Allen, Resonance, FIITJEE, Aakash Institute, and Narayana for JEE/NEET. For other subjects, local coaching centers with experienced teachers often work better. Look for centers with good faculty, comprehensive study materials, regular tests, and doubt-clearing sessions. Online platforms like Physics Wallah, Unacademy, and Vedantu are cost-effective alternatives with quality content.\n",
            
            "Question: How do I choose between online and offline tuition for {}?\nAnswer: For {}, online tuition offers flexibility, recorded lectures, and cost savings. It's ideal for self-motivated students with good internet connectivity. Offline tuition provides personal interaction, immediate doubt resolution, and structured environment. It works better for students who need constant monitoring. Consider hybrid approach - use online resources for concept building and offline classes for doubt clearing and practice.\n",
            
            "Question: Is home tuition better than coaching centers for {}?\nAnswer: Home tuition for {} provides personalized attention and customized pace. It's beneficial for students with specific weak areas or those preparing for board exams. Coaching centers offer competitive environment, comprehensive materials, and peer learning. They're better for competitive exam preparation. Your choice should depend on learning goals, budget, and whether you need individual attention or group dynamics.\n",
            
            "Question: How much should I spend on tuition for {}?\nAnswer: Tuition fees for {} vary widely based on location and type. Coaching institutes charge ₹50,000-₹2,00,000 annually for competitive exam preparation. Home tuitions range from ₹5,000-₹20,000 monthly depending on expertise. Online platforms are most affordable at ₹5,000-₹30,000 annually. Don't overspend on expensive coaching if quality free resources are available. Focus on consistent effort rather than costly tuitions.\n"
        ]
        
        for _ in range(count):
            template = random.choice(templates)
            subject = random.choice(self.subjects["school"] + list(self.exams.keys()))
            self.dataset.append(template.format(subject, subject))
            self.line_count += template.count('\n')

    def generate_college_queries(self, count: int):
        templates = [
            "Question: How do I get admission to {}?\nAnswer: Admission to {} requires excellent performance in entrance exams and academics. For IITs, crack JEE Advanced with top ranks. Maintain good scores in Class 12 boards. For other premier institutes, perform well in respective entrance tests. Prepare strategically for 12-18 months. Join coaching if needed but supplement with self-study. Stay updated with admission criteria and cutoffs. Apply through JoSAA counseling for IITs/NITs. Consider multiple colleges to keep options open.\n",
            
            "Question: What is the cutoff for {} in {}?\nAnswer: Cutoff for {} in {} varies yearly based on exam difficulty and seat availability. Generally, you need ranks within top 10,000 for IITs, top 50,000 for NITs in popular branches. Check previous year cutoffs on official websites. Cutoffs are lower for reserved categories. General category students need higher percentiles. Prepare to score 95+ percentile for top colleges. Focus on consistent preparation rather than predicting cutoffs.\n",
            
            "Question: Which branch should I choose in {}?\nAnswer: Branch selection in {} should align with your interests and career goals. Computer Science offers high placements but is highly competitive. Core branches like Mechanical, Civil need passion as initial placements are moderate. Consider future scope, industry demand, and your aptitude. Don't choose solely based on placements. Research about curriculum and career paths. Talk to seniors and alumni. A good branch in average college can be better than average branch in top college.\n",
            
            "Question: What is campus life like at {}?\nAnswer: Campus life at {} offers excellent infrastructure, diverse cultural activities, and strong peer network. Academic rigor is high with challenging coursework. Multiple technical and cultural clubs provide holistic development. Hostels have good facilities with vibrant community. Placement opportunities are strong with top companies visiting. Sports facilities are well-maintained. Inter-college festivals add to experience. Balance academics with extracurriculars for complete development.\n",
            
            "Question: What are the placement statistics of {}?\nAnswer: {} consistently shows strong placement records with top companies recruiting. Average package ranges from ₹8-15 lakhs for engineering branches. Computer Science sees highest packages of ₹20-40 lakhs. Core branches have improving trends. Many students pursue higher studies or startups. Placement percentage typically exceeds 85-90%. Top recruiters include Microsoft, Google, Amazon, Goldman Sachs. Focus on building skills alongside academics for better opportunities.\n"
        ]
        
        for _ in range(count):
            template = random.choice(templates)
            college = random.choice(self.indian_colleges)
            if template.count('{}') == 1:
                self.dataset.append(template.format(college, college))
            elif template.count('{}') == 2:
                branch = random.choice(["Computer Science", "Mechanical", "Electrical", "Civil", "Electronics"])
                self.dataset.append(template.format(branch, college, branch, college))
            self.line_count += template.count('\n')

    def generate_subject_explanations(self, count: int):
        for _ in range(count):
            category = random.choice(list(self.subjects.keys()))
            subject = random.choice(self.subjects[category])
            
            explanation = f"Question: Explain the concept of {subject}.\nAnswer: {subject} is a fundamental topic in {category} education. "
            
            if category == "school":
                explanation += f"It builds critical thinking and analytical skills. Students learn through theory, practical applications, and problem-solving. Regular practice with NCERT textbooks and reference materials strengthens understanding. This concept appears frequently in board exams and competitive tests. Understanding the basics thoroughly is crucial before moving to advanced topics.\n"
            elif category == "engineering":
                explanation += f"This subject is essential for developing technical expertise. It involves both theoretical knowledge and practical implementation. Students work on projects and assignments to gain hands-on experience. Industry applications include software development, system design, and optimization. Strong foundation in this area opens multiple career opportunities.\n"
            elif category == "medical":
                explanation += f"This forms the core of medical education and clinical practice. Students must memorize key concepts while understanding underlying mechanisms. Practical sessions and clinical rotations reinforce learning. This knowledge is tested in NEET PG and other medical entrance exams. Thorough understanding is critical for patient care and diagnosis.\n"
            else:
                explanation += f"This is crucial for business and management education. It combines theoretical frameworks with real-world case studies. Students develop strategic thinking and decision-making skills. Applications span across industries and organizational contexts. MBA programs emphasize this through interactive sessions and group projects.\n"
            
            self.dataset.append(explanation)
            self.line_count += explanation.count('\n')

    def generate_homework_help(self, count: int):
        templates = [
            "Question: Can you help me with my {} homework?\nAnswer: I can guide you through your {} homework. First, read the question carefully and identify what's being asked. Break down the problem into smaller parts. Review relevant concepts from your textbook or notes. Try solving step-by-step. If stuck, check solved examples. I can explain concepts but you should attempt solving yourself for better learning. Practice similar problems to strengthen understanding.\n",
            
            "Question: I don't understand this {} problem. Can you explain?\nAnswer: Let me help you understand this {} problem. Start by identifying given information and what needs to be found. Review the formula or theorem applicable here. Draw diagrams if needed for visualization. Work through each step logically. Common mistakes include calculation errors or misunderstanding concepts. Practice more problems of this type. If still confused, ask your teacher or join study group for different perspectives.\n",
            
            "Question: How do I complete my {} assignment on time?\nAnswer: To complete {} assignment on time, start early rather than procrastinating. Break assignment into smaller tasks with deadlines. Allocate specific time slots daily for working on it. Gather all required materials and resources beforehand. Focus on quality rather than rushing at last minute. Take short breaks to maintain concentration. Review work before submission. Time management and discipline are key to timely completion.\n"
        ]
        
        for _ in range(count):
            template = random.choice(templates)
            subject = random.choice(self.subjects["school"])
            self.dataset.append(template.format(subject, subject))
            self.line_count += template.count('\n')

    def generate_concept_clarifications(self, count: int):
        for _ in range(count):
            subject = random.choice(self.subjects["school"])
            topic = random.choice(["theorem", "principle", "law", "concept", "formula"])
            
            clarification = f"Question: I'm confused about {topic}s in {subject}. Can you clarify?\nAnswer: Understanding {topic}s in {subject} requires systematic approach. Start with definitions and fundamental principles. Study derivations to see how formulas are developed. Practice applying them to various problems. Create a reference sheet with all important {topic}s. Use mnemonics or visual aids for memorization. Understand when and how to use each {topic}. Regular revision prevents confusion. Solve previous year questions to see practical applications. Don't hesitate to ask teachers for additional explanation.\n"
            
            self.dataset.append(clarification)
            self.line_count += clarification.count('\n')

    def generate_roadmaps(self, target_lines: int):
        print(f"Generating roadmaps: {target_lines} lines...")
        categories = [
            self.generate_career_roadmaps,
            self.generate_exam_preparation_roadmaps,
            self.generate_skill_development_roadmaps,
            self.generate_college_preparation_roadmaps
        ]
        
        lines_per_category = target_lines // len(categories)
        for category_func in categories:
            category_func(lines_per_category)

    def generate_career_roadmaps(self, count: int):
        careers = [
            ("Software Engineer", ["Learn programming basics", "Master data structures and algorithms", 
             "Build projects and contribute to open source", "Prepare for technical interviews", "Apply for internships"]),
            ("Data Scientist", ["Learn statistics and probability", "Master Python and R", "Study machine learning algorithms",
             "Work on real-world datasets", "Build portfolio projects"]),
            ("Doctor", ["Excel in NEET preparation", "Join MBBS in good medical college", "Complete internship",
             "Choose specialization for MD/MS", "Practice and gain clinical experience"]),
            ("IAS Officer", ["Graduate in any discipline", "Start UPSC preparation early", "Cover entire syllabus systematically",
             "Practice answer writing", "Clear Prelims, Mains, and Interview"]),
            ("MBA Graduate", ["Complete undergraduate degree with good grades", "Prepare for CAT/GMAT", 
             "Join top B-school", "Gain internship experience", "Specialize in domain of interest"]),
            ("Mechanical Engineer", ["Strong foundation in physics and mathematics", "Join good engineering college",
             "Learn CAD software", "Do internships in core companies", "Consider higher studies or job"]),
            ("Civil Services", ["Choose optional subject wisely", "Study NCERT thoroughly", "Read newspapers daily",
             "Join test series", "Practice answer writing regularly"]),
            ("Chartered Accountant", ["Complete 10+2 with commerce", "Clear CA Foundation", "Article ship training",
             "Clear CA Intermediate and Final", "Gain practical experience"])
        ]
        
        for _ in range(count):
            career, steps = random.choice(careers)
            
            roadmap = f"Question: What is the roadmap to become a {career}?\nAnswer: To become a {career}, follow this comprehensive roadmap:\n\n"
            
            for i, step in enumerate(steps, 1):
                roadmap += f"Step {i}: {step}\n"
                if i == 1:
                    roadmap += f"Duration: 6-12 months. Focus on building strong fundamentals. Use quality learning resources and practice regularly.\n\n"
                elif i == len(steps):
                    roadmap += f"Duration: Ongoing. Continuous learning and adaptation to industry changes is crucial. Network with professionals and stay updated.\n\n"
                else:
                    roadmap += f"Duration: 1-2 years. Dedicate consistent effort and track your progress. Seek mentorship when needed.\n\n"
            
            roadmap += f"Total timeline: 4-8 years depending on chosen path. Success requires dedication, continuous learning, and persistence. Adapt roadmap based on opportunities and interests.\n"
            
            self.dataset.append(roadmap)
            self.line_count += roadmap.count('\n')

    def generate_exam_preparation_roadmaps(self, count: int):
        for _ in range(count):
            exam_name = random.choice(list(self.exams.keys()))
            exam_info = self.exams[exam_name]
            
            months = random.choice([6, 12, 18, 24])
            
            roadmap = f"Question: How should I prepare for {exam_name} in {months} months?\nAnswer: Here's a detailed {months}-month preparation roadmap for {exam_name}:\n\n"
            
            if months >= 18:
                roadmap += "Phase 1 (Months 1-8): Foundation Building\n"
                roadmap += f"- Complete entire syllabus of {', '.join(exam_info['subjects'])}\n"
                roadmap += "- Study from standard textbooks and make comprehensive notes\n"
                roadmap += "- Focus on understanding concepts rather than speed\n"
                roadmap += "- Solve chapter-wise problems after completing each topic\n\n"
                
                roadmap += "Phase 2 (Months 9-14): Advanced Preparation\n"
                roadmap += "- Revise all topics and strengthen weak areas\n"
                roadmap += "- Practice previous year questions subject-wise\n"
                roadmap += "- Start taking topic-wise tests\n"
                roadmap += "- Join test series for regular assessment\n\n"
                
                roadmap += "Phase 3 (Months 15-18): Final Sprint\n"
                roadmap += "- Take full-length mock tests regularly\n"
                roadmap += "- Analyze performance and work on mistakes\n"
                roadmap += "- Quick revision using notes and formula sheets\n"
                roadmap += "- Focus on time management and exam strategy\n"
            
            elif months >= 12:
                roadmap += "Phase 1 (Months 1-6): Core Preparation\n"
                roadmap += f"- Complete syllabus of {', '.join(exam_info['subjects'])} systematically\n"
                roadmap += "- Make concise notes for quick revision\n"
                roadmap += "- Practice numerical problems daily\n\n"
                
                roadmap += "Phase 2 (Months 7-10): Practice & Testing\n"
                roadmap += "- Solve previous year papers extensively\n"
                roadmap += "- Take regular mock tests\n"
                roadmap += "- Work on speed and accuracy\n\n"
                
                roadmap += "Phase 3 (Months 11-12): Final Revision\n"
                roadmap += "- Revise weak topics thoroughly\n"
                roadmap += "- Take full-length mocks in exam conditions\n"
                roadmap += "- Stay calm and maintain health\n"
            
            else:
                roadmap += "Intensive 6-Month Plan:\n"
                roadmap += "- Months 1-3: Cover entire syllabus quickly\n"
                roadmap += "- Months 4-5: Practice and mock tests\n"
                roadmap += "- Month 6: Revision and final preparation\n"
                roadmap += "- Study 8-10 hours daily with focus\n"
                roadmap += "- Join crash courses if needed\n"
            
            roadmap += f"\nKey Tips: Stay consistent, maintain healthy routine, take regular breaks, and believe in yourself. {exam_name} difficulty is {exam_info['difficulty']}, so prepare accordingly.\n"
            
            self.dataset.append(roadmap)
            self.line_count += roadmap.count('\n')

    def generate_skill_development_roadmaps(self, count: int):
        skills = [
            "Machine Learning", "Web Development", "Mobile App Development", "Data Analysis",
            "Cybersecurity", "Cloud Computing", "DevOps", "Digital Marketing",
            "Graphic Design", "Content Writing", "Video Editing", "Public Speaking"
        ]
        
        for _ in range(count):
            skill = random.choice(skills)
            
            roadmap = f"Question: What is the learning roadmap for {skill}?\nAnswer: Complete roadmap to master {skill}:\n\n"
            roadmap += "Beginner Level (0-3 months):\n"
            roadmap += f"- Understand fundamentals and basic concepts of {skill}\n"
            roadmap += "- Complete introductory online courses from Coursera, Udemy, or YouTube\n"
            roadmap += "- Work on small practice projects\n"
            roadmap += "- Join online communities and forums\n\n"
            
            roadmap += "Intermediate Level (3-9 months):\n"
            roadmap += f"- Deep dive into advanced topics of {skill}\n"
            roadmap += "- Build 3-5 portfolio projects showcasing your skills\n"
            roadmap += "- Contribute to open-source projects if applicable\n"
            roadmap += "- Network with professionals in the field\n\n"
            
            roadmap += "Advanced Level (9-18 months):\n"
            roadmap += f"- Master specialized areas within {skill}\n"
            roadmap += "- Work on complex real-world projects\n"
            roadmap += "- Consider freelancing or internships for experience\n"
            roadmap += "- Stay updated with latest trends and technologies\n\n"
            
            roadmap += "Expert Level (18+ months):\n"
            roadmap += "- Become thought leader by sharing knowledge\n"
            roadmap += "- Mentor others and contribute to community\n"
            roadmap += "- Pursue certifications if relevant\n"
            roadmap += "- Continuously evolve with industry changes\n\n"
            
            roadmap += f"Resources: Online courses, documentation, YouTube tutorials, books, and hands-on practice. Consistency is key to mastering {skill}.\n"
            
            self.dataset.append(roadmap)
            self.line_count += roadmap.count('\n')

    def generate_college_preparation_roadmaps(self, count: int):
        for _ in range(count):
            college = random.choice(self.indian_colleges)
            exam = "JEE Advanced" if "IIT" in college else "NEET" if "AIIMS" in college else "CAT" if "IIM" in college else "entrance exam"
            
            roadmap = f"Question: Complete roadmap to get into {college}?\nAnswer: Comprehensive preparation roadmap for {college} admission:\n\n"
            
            roadmap += "Class 11 (Foundation Year):\n"
            roadmap += f"- Build strong foundation in {exam} subjects\n"
            roadmap += "- Maintain 90%+ in school exams\n"
            roadmap += "- Start coaching or self-study systematically\n"
            roadmap += "- Complete 70-80% of Class 11 and 12 syllabus\n\n"
            
            roadmap += "Class 12 (Preparation Year):\n"
            roadmap += "- Balance board exam and entrance preparation\n"
            roadmap += "- Complete entire syllabus by December\n"
            roadmap += "- Take regular mock tests\n"
            roadmap += "- Revise Class 11 concepts thoroughly\n\n"
            
            roadmap += "Final Months:\n"
            roadmap += f"- Intensive practice for {exam}\n"
            roadmap += "- Solve 20+ years previous papers\n"
            roadmap += "- Analyze mistakes and improve weak areas\n"
            roadmap += "- Stay healthy and manage stress\n\n"
            
            roadmap += f"Target: Rank within top 500-1000 for {college} in popular branches. Requires 10-12 hours daily study, smart preparation strategy, and unwavering dedication.\n"
            
            self.dataset.append(roadmap)
            self.line_count += roadmap.count('\n')

    def generate_mental_health_conversations(self, target_lines: int):
        print(f"Generating mental health conversations: {target_lines} lines...")
        categories = [
            self.generate_stress_management,
            self.generate_anxiety_support,
            self.generate_motivation_content,
            self.generate_study_burnout,
            self.generate_peer_pressure,
            self.generate_failure_coping
        ]
        
        lines_per_category = target_lines // len(categories)
        for category_func in categories:
            category_func(lines_per_category)

    def generate_stress_management(self, count: int):
        templates = [
            "Question: I'm feeling overwhelmed with my studies. What should I do?\nAnswer: Feeling overwhelmed is common among students. First, acknowledge your feelings without judgment. Break your study load into smaller, manageable tasks. Create a realistic daily schedule with proper breaks. Practice deep breathing or meditation for 10 minutes daily. Talk to someone you trust about your feelings. Exercise regularly to release stress. Ensure 7-8 hours of sleep. Remember, it's okay to ask for help from teachers, parents, or counselors. Your mental health is as important as academic success.\n",
            
            "Question: How do I manage exam stress?\nAnswer: Exam stress is natural but manageable. Start preparation early to avoid last-minute panic. Create a study timetable and stick to it. Take regular 10-minute breaks every hour. Stay hydrated and eat nutritious food. Avoid caffeine overload. Practice relaxation techniques like progressive muscle relaxation. Exercise daily even if briefly. Sleep well before exams. Avoid comparing with peers. Remember that one exam doesn't define your future. If anxiety persists, consider speaking with a counselor.\n",
            
            "Question: I can't concentrate on studies due to stress.\nAnswer: Stress significantly impacts concentration. First, identify stress sources - academic pressure, family expectations, or personal issues. Address these systematically. Create a dedicated study space free from distractions. Use Pomodoro technique - 25 minutes focused study, 5 minutes break. Practice mindfulness meditation to improve focus. Ensure adequate sleep and nutrition. Physical exercise boosts concentration. If problems persist beyond 2 weeks, consult a mental health professional. Your well-being comes first.\n",
            
            "Question: My parents' expectations are causing me stress.\nAnswer: Parental expectations often create pressure. Have an honest, calm conversation with your parents about your feelings. Help them understand your capabilities and limitations. Set realistic goals together. Show them your efforts and progress. Remember, parents want best for you but may not realize the pressure. If direct communication is difficult, involve a trusted teacher or relative. Focus on doing your personal best rather than meeting unrealistic expectations. Seek counseling if the stress becomes unmanageable.\n"
        ]
        
        for _ in range(count):
            self.dataset.append(random.choice(templates))
            self.line_count += random.choice(templates).count('\n')

    def generate_anxiety_support(self, count: int):
        scenarios = [
            ("exam anxiety", "performance worries", "breathing exercises and positive self-talk"),
            ("social anxiety", "peer interactions", "gradual exposure and social skills practice"),
            ("future anxiety", "career uncertainty", "planning and focusing on present actions"),
            ("performance anxiety", "academic pressure", "realistic goal-setting and self-compassion")
        ]
        
        for _ in range(count):
            anxiety_type, context, solution = random.choice(scenarios)
            
            content = f"Question: I have {anxiety_type} related to {context}. How do I cope?\nAnswer: {anxiety_type.title()} is a common challenge. Understanding that anxiety is your mind's way of trying to protect you can help. For {context}, start by identifying specific triggers. Practice {solution} regularly. Challenge negative thoughts with evidence-based thinking. Remember past successes when you overcame similar situations. Maintain a journal to track anxiety patterns and what helps. Stay physically active and maintain regular sleep schedule. Consider professional help if anxiety interferes with daily life. Remember, seeking help is a sign of strength, not weakness. You're not alone in this.\n"
            
            self.dataset.append(content)
            self.line_count += content.count('\n')

    def generate_motivation_content(self, count: int):
        templates = [
            "Question: I've lost motivation to study. How do I get it back?\nAnswer: Loss of motivation is temporary and recoverable. First, revisit why you started - your goals and dreams. Break large goals into small achievable targets. Celebrate small wins to build momentum. Change your study environment or routine for freshness. Study with motivated peers for inspiration. Visualize your success regularly. Remember that motivation follows action, not vice versa. Start with just 15 minutes of study. Take care of physical health - exercise, sleep, nutrition. If demotivation persists, explore if there are deeper issues like depression or burnout that need professional attention.\n",
            
            "Question: How do I stay motivated during long-term preparation like JEE or NEET?\nAnswer: Long-term motivation requires sustainable strategies. Set both short-term weekly goals and long-term targets. Track your progress visually with charts. Take one day off weekly for complete rest. Engage in hobbies to maintain balance. Connect with aspirants and share experiences. Read success stories of seniors. Remember your 'why' - write it down and review regularly. Accept that some days will be low productivity days. Focus on consistency over perfection. Reward yourself after achieving milestones. Stay connected with family and friends. Self-care isn't selfish - it's necessary for sustained performance.\n",
            
            "Question: I feel like giving up on my goals. What should I do?\nAnswer: Feeling like giving up is a signal to pause and reassess, not necessarily to quit. Take a short break to gain perspective. Reflect on your journey so far and progress made. Talk to a mentor or counselor about your feelings. Ask yourself: Am I overwhelmed by goal size? Do I need better strategy? Are external factors affecting me? Adjust your approach if needed, but don't abandon goals due to temporary setbacks. Remember why you started. Look at how far you've come. Many successful people faced similar moments. If the goal truly doesn't align with you anymore, it's okay to pivot. But don't quit in a moment of weakness.\n"
        ]
        
        for _ in range(count):
            self.dataset.append(random.choice(templates))
            self.line_count += random.choice(templates).count('\n')

    def generate_study_burnout(self, count: int):
        for _ in range(count):
            content = "Question: I think I'm experiencing study burnout. What are the signs and how do I recover?\nAnswer: Burnout is serious and needs attention. Signs include: chronic exhaustion, decreased performance despite effort, lack of motivation, irritability, difficulty concentrating, physical symptoms like headaches, and feeling detached. Recovery requires:\n\n"
            content += "1. Immediate Action: Take a complete break for 3-7 days. No studying, no guilt.\n"
            content += "2. Reassess: Evaluate your study schedule. Are you overworking? Reduce daily study hours.\n"
            content += "3. Self-Care: Prioritize sleep (8 hours), exercise (30 min daily), healthy eating.\n"
            content += "4. Reconnect: Spend time with friends and family. Pursue hobbies.\n"
            content += "5. Set Boundaries: Learn to say no. Not every opportunity is worth pursuing.\n"
            content += "6. Seek Support: Talk to counselor, teacher, or mental health professional.\n"
            content += "7. Restructure: Create a balanced schedule with buffer time and regular breaks.\n\n"
            content += "Remember, burnout happens when we push beyond our limits repeatedly. Recovery takes time. Be patient and kind to yourself. Your worth isn't determined by academic performance alone.\n"
            
            self.dataset.append(content)
            self.line_count += content.count('\n')

    def generate_peer_pressure(self, count: int):
        templates = [
            "Question: All my friends are studying 12 hours daily. Should I do the same?\nAnswer: Don't let peer comparison drive your study schedule. Everyone has different learning speeds, efficiency levels, and circumstances. What matters is quality over quantity. Some students genuinely need more time, others learn faster. Studying 12 hours daily isn't sustainable for most people and often leads to burnout. Focus on your optimal study hours with full concentration. If you're productive in 6-8 hours of focused study, that's perfect. Track your own progress, not others'. Remember, social media and peer conversations often exaggerate study hours. Do what works for you, not what others claim to do.\n",
            
            "Question: My classmates are going to expensive coaching. Am I at disadvantage?\nAnswer: Expensive coaching doesn't guarantee success. Many toppers prepare through self-study or affordable resources. What matters is consistent effort and smart preparation. Free resources like NPTEL, Khan Academy, YouTube channels, and library books are excellent. Focus on understanding concepts thoroughly rather than collecting coaching materials. Expensive coaching provides structure and peer environment, but self-discipline can achieve same results. If you're worried, join affordable online platforms. Your dedication and strategy matter more than coaching fees. Many successful people succeeded without expensive coaching. Believe in your preparation.\n"
        ]
        
        for _ in range(count):
            self.dataset.append(random.choice(templates))
            self.line_count += random.choice(templates).count('\n')

    def generate_failure_coping(self, count: int):
        for _ in range(count):
            scenarios = [
                ("failed in board exams", "one exam doesn't define your future", "supplementary exams and improvement options"),
                ("didn't clear JEE/NEET", "multiple attempts are common", "drop year or alternate career paths"),
                ("low marks in semester", "one semester can be improved", "focused study for upcoming exams"),
                ("rejected from dream college", "many paths lead to success", "making the most of current opportunity")
            ]
            
            scenario, perspective, action = random.choice(scenarios)
            
            content = f"Question: I {scenario}. I feel like a failure.\nAnswer: First, understand that {perspective}. Your feelings of disappointment are valid, but they don't reflect your worth or potential. Failure is a learning opportunity, not an endpoint. Here's how to cope:\n\n"
            content += "1. Allow yourself to feel: It's okay to be sad, angry, or disappointed. Don't suppress emotions.\n"
            content += "2. Reach out: Talk to family, friends, or counselor. You don't have to go through this alone.\n"
            content += "3. Avoid self-blame: Analyze what went wrong without harsh self-criticism. Be objective.\n"
            content += f"4. Explore options: Consider {action}. Multiple pathways exist to success.\n"
            content += "5. Learn lessons: What can you do differently? Better preparation? Time management? Stress handling?\n"
            content += "6. Set new goals: After processing emotions, set realistic next steps. Break them into achievable tasks.\n"
            content += "7. Celebrate effort: Acknowledge the hard work you put in, regardless of outcome.\n\n"
            content += "Remember: Many successful people faced similar setbacks. Steve Jobs was fired from Apple. Abdul Kalam didn't clear air force selection. Your story isn't over - this is just one chapter. Keep moving forward.\n"
            
            self.dataset.append(content)
            self.line_count += content.count('\n')

    def generate_exam_preparation_content(self, target_lines: int):
        print(f"Generating exam preparation content: {target_lines} lines...")
        categories = [
            self.generate_exam_strategies,
            self.generate_exam_requirements,
            self.generate_time_management,
            self.generate_revision_techniques,
            self.generate_mock_test_guidance
        ]
        
        lines_per_category = target_lines // len(categories)
        for category_func in categories:
            category_func(lines_per_category)

    def generate_exam_strategies(self, count: int):
        for _ in range(count):
            exam_name = random.choice(list(self.exams.keys()))
            exam_info = self.exams[exam_name]
            
            strategy = f"Question: What is the best strategy to crack {exam_name}?\nAnswer: To crack {exam_name} successfully, follow this comprehensive strategy:\n\n"
            strategy += "1. Understand Exam Pattern:\n"
            strategy += f"- Study {exam_name} syllabus thoroughly\n"
            strategy += "- Analyze previous 5-10 years question papers\n"
            strategy += "- Identify frequently asked topics and question types\n"
            strategy += f"- Understand marking scheme and difficulty level ({exam_info['difficulty']})\n\n"
            
            strategy += "2. Preparation Strategy:\n"
            strategy += f"- Focus on {', '.join(exam_info['subjects'])}\n"
            strategy += "- Create subject-wise study plan with deadlines\n"
            strategy += "- Use standard textbooks and reference materials\n"
            strategy += "- Make concise notes for quick revision\n"
            strategy += "- Practice numerical problems daily\n\n"
            
            strategy += "3. Practice and Testing:\n"
            strategy += "- Solve previous year papers in timed conditions\n"
            strategy += "- Take full-length mock tests weekly\n"
            strategy += "- Analyze mistakes and work on weak areas\n"
            strategy += "- Improve speed and accuracy simultaneously\n"
            strategy += "- Join online test series for regular assessment\n\n"
            
            strategy += "4. Final Preparation:\n"
            strategy += "- Revise all topics multiple times\n"
            strategy += "- Focus on high-weightage topics\n"
            strategy += "- Practice time management strategies\n"
            strategy += "- Stay calm and confident\n"
            strategy += "- Take care of health and sleep\n\n"
            
            strategy += f"Success in {exam_name} requires consistent effort, smart preparation, and positive mindset. Start early and stay focused throughout your preparation journey.\n"
            
            self.dataset.append(strategy)
            self.line_count += strategy.count('\n')

    def generate_exam_requirements(self, count: int):
        for _ in range(count):
            exam_name = random.choice(list(self.exams.keys()))
            exam_info = self.exams[exam_name]
            
            requirements = f"Question: What are the requirements and eligibility for {exam_name}?\nAnswer: Complete information about {exam_name} requirements:\n\n"
            
            if "JEE" in exam_name:
                requirements += "Eligibility Criteria:\n"
                requirements += "- Minimum 75% in Class 12 boards (65% for SC/ST)\n"
                requirements += "- Age limit: Born on or after October 1, 2000 (relaxation for reserved categories)\n"
                requirements += "- Maximum attempts: 6 attempts over 3 consecutive years\n"
                requirements += "- Must have Physics, Chemistry, and Mathematics in 12th\n\n"
                requirements += "Exam Pattern:\n"
                requirements += "- Duration: 3 hours\n"
                requirements += "- Mode: Computer-based test\n"
                requirements += "- Questions: Multiple choice and numerical type\n"
                requirements += "- Marking: +4 for correct, -1 for incorrect\n\n"
            elif "NEET" in exam_name:
                requirements += "Eligibility Criteria:\n"
                requirements += "- Minimum age: 17 years as of December 31\n"
                requirements += "- Maximum age: Upper age limit removed\n"
                requirements += "- Must have Physics, Chemistry, and Biology in 12th\n"
                requirements += "- Minimum 50% marks in PCB (40% for SC/ST/OBC)\n\n"
                requirements += "Exam Pattern:\n"
                requirements += "- Duration: 3 hours 20 minutes\n"
                requirements += "- Mode: Pen and paper based\n"
                requirements += "- 200 questions (180 to be attempted)\n"
                requirements += "- Marking: +4 for correct, -1 for incorrect\n\n"
            elif "GATE" in exam_name:
                requirements += "Eligibility Criteria:\n"
                requirements += "- Bachelor's degree in Engineering/Technology/Architecture\n"
                requirements += "- Or Master's degree in relevant science subject\n"
                requirements += "- Final year students can also apply\n"
                requirements += "- No age limit\n\n"
                requirements += "Exam Pattern:\n"
                requirements += "- Duration: 3 hours\n"
                requirements += "- Mode: Computer-based test\n"
                requirements += "- Multiple choice and numerical questions\n"
                requirements += "- Marking scheme varies by question type\n\n"
            elif "CAT" in exam_name:
                requirements += "Eligibility Criteria:\n"
                requirements += "- Bachelor's degree in any discipline\n"
                requirements += "- Minimum 50% marks (45% for SC/ST/PwD)\n"
                requirements += "- Final year students can apply\n"
                requirements += "- No age limit or number of attempts restriction\n\n"
                requirements += "Exam Pattern:\n"
                requirements += "- Duration: 2 hours (40 minutes per section)\n"
                requirements += "- Three sections: VARC, DILR, QA\n"
                requirements += "- 66 questions total\n"
                requirements += "- Marking: +3 for correct, -1 for incorrect\n\n"
            else:
                requirements += "Eligibility Criteria:\n"
                requirements += "- Check official notification for specific requirements\n"
                requirements += "- Educational qualification as per exam guidelines\n"
                requirements += "- Age limit if applicable\n\n"
                requirements += "Exam Pattern:\n"
                requirements += "- Duration and mode as per official notification\n"
                requirements += "- Question types and marking scheme varies\n\n"
            
            requirements += "Important Documents:\n"
            requirements += "- Class 10 and 12 mark sheets\n"
            requirements += "- Graduation degree (if applicable)\n"
            requirements += "- Category certificate (if applicable)\n"
            requirements += "- ID proof and photographs\n"
            requirements += "- Disability certificate (if applicable)\n\n"
            
            requirements += f"Application Process: Apply online through official {exam_name} website. Keep all documents ready. Pay application fee through online mode. Take printout of confirmation page for future reference.\n"
            
            self.dataset.append(requirements)
            self.line_count += requirements.count('\n')

    def generate_time_management(self, count: int):
        templates = [
            "Question: How do I manage time during exam preparation?\nAnswer: Effective time management is crucial for exam success. Create a realistic timetable allocating time to each subject based on difficulty and syllabus coverage. Use time blocking technique - dedicate specific hours to specific subjects. Follow 50-10 rule: study 50 minutes, break 10 minutes. Identify your peak productivity hours and schedule difficult subjects then. Use tools like Pomodoro technique for focused study. Avoid multitasking during study hours. Set daily, weekly, and monthly targets. Track time spent on each activity. Minimize distractions by keeping phone away. Include buffer time for unexpected situations. Review and adjust timetable weekly based on progress. Balance study with adequate sleep, exercise, and leisure. Remember, consistency matters more than long study hours.\n",
            
            "Question: How should I allocate time between different subjects?\nAnswer: Time allocation depends on multiple factors: subject difficulty for you, syllabus weightage, current preparation level, and time available. Analyze each subject's importance in exam. Allocate 40% time to difficult subjects, 30% to moderate, and 30% to easier ones. Within each subject, identify high-weightage topics and prioritize them. Don't neglect easier subjects completely - maintain regular practice. If preparing for multiple exams, identify overlapping topics and optimize study. Reallocate time monthly based on progress. Weaker subjects need more initial time but reduce as you improve. Create flexibility in schedule for unexpected challenges. Use weekends for comprehensive revision. Quality focused study of 6-8 hours is better than 12 unfocused hours.\n",
            
            "Question: How do I manage time during actual exam?\nAnswer: Time management during exam determines success. Before starting, quickly scan entire paper to understand difficulty level and question distribution. Allocate time to each section based on marks. Start with questions you know well to build confidence and secure those marks. Don't spend too much time on single difficult question - mark it and move on. Keep checking watch every 15-20 minutes to track progress. Reserve last 10-15 minutes for revision and attempting marked questions. For objective exams, manage time per question - typically 2-3 minutes per question. If stuck, make educated guess and move forward. In subjective exams, write concise, relevant answers. Practice time management in mock tests extensively. Develop exam rhythm through regular practice. Stay calm if running short of time - prioritize remaining high-marks questions.\n"
        ]
        
        for _ in range(count):
            self.dataset.append(random.choice(templates))
            self.line_count += random.choice(templates).count('\n')

    def generate_revision_techniques(self, count: int):
        techniques = [
            ("Spaced Repetition", "review material at increasing intervals", "enhances long-term retention"),
            ("Active Recall", "actively test yourself instead of passive reading", "strengthens memory connections"),
            ("Feynman Technique", "explain concepts in simple language", "ensures deep understanding"),
            ("Mind Mapping", "create visual connections between topics", "aids in seeing big picture"),
            ("Practice Testing", "solve problems under exam conditions", "improves speed and accuracy"),
            ("Interleaving", "mix different subjects/topics in single session", "enhances learning and retention")
        ]
        
        for _ in range(count):
            technique, method, benefit = random.choice(techniques)
            
            content = f"Question: What is {technique} and how do I use it for revision?\nAnswer: {technique} is a powerful revision strategy where you {method}. This technique is effective because it {benefit}.\n\n"
            content += f"How to implement {technique}:\n"
            
            if technique == "Spaced Repetition":
                content += "- First revision: Same day after learning\n"
                content += "- Second revision: After 1 day\n"
                content += "- Third revision: After 3 days\n"
                content += "- Fourth revision: After 1 week\n"
                content += "- Fifth revision: After 2 weeks\n"
                content += "- Continue at increasing intervals\n"
                content += "Use flashcard apps like Anki for automation\n\n"
            elif technique == "Active Recall":
                content += "- Close your notes/book\n"
                content += "- Try to recall key points from memory\n"
                content += "- Write down everything you remember\n"
                content += "- Check against original material\n"
                content += "- Note what you missed\n"
                content += "- Repeat the process\n\n"
            elif technique == "Feynman Technique":
                content += "- Choose a concept to learn\n"
                content += "- Explain it as if teaching a child\n"
                content += "- Identify gaps in your understanding\n"
                content += "- Go back and re-learn those parts\n"
                content += "- Simplify explanations further\n"
                content += "- Use analogies and examples\n\n"
            elif technique == "Mind Mapping":
                content += "- Write central topic in middle\n"
                content += "- Draw branches for subtopics\n"
                content += "- Add details to each branch\n"
                content += "- Use colors and images\n"
                content += "- Show connections between concepts\n"
                content += "- Review and update regularly\n\n"
            
            content += f"Best used for: {technique} works excellently for conceptual subjects and long-term retention. Combine with other techniques for comprehensive revision. Regular practice makes this technique second nature.\n"
            
            self.dataset.append(content)
            self.line_count += content.count('\n')

    def generate_mock_test_guidance(self, count: int):
        for _ in range(count):
            exam_name = random.choice(list(self.exams.keys()))
            
            guidance = f"Question: How should I approach mock tests for {exam_name}?\nAnswer: Mock tests are crucial for {exam_name} preparation. Follow this comprehensive approach:\n\n"
            guidance += "Before Mock Test:\n"
            guidance += "- Complete at least 60-70% syllabus before starting mocks\n"
            guidance += "- Choose test series that match actual exam pattern\n"
            guidance += "- Schedule mock tests on same day and time as actual exam\n"
            guidance += "- Keep all materials ready as per exam requirements\n"
            guidance += "- Ensure distraction-free environment\n\n"
            
            guidance += "During Mock Test:\n"
            guidance += "- Follow exact exam timing and rules\n"
            guidance += "- Don't use phone or reference materials\n"
            guidance += "- Practice time management strategies\n"
            guidance += "- Mark difficult questions for later review\n"
            guidance += "- Simulate actual exam conditions completely\n\n"
            
            guidance += "After Mock Test:\n"
            guidance += "- Analyze performance immediately while fresh\n"
            guidance += "- Review all incorrect answers thoroughly\n"
            guidance += "- Understand why you got questions wrong\n"
            guidance += "- Note topics needing more practice\n"
            guidance += "- Track improvement across multiple mocks\n"
            guidance += "- Create weak area revision plan\n\n"
            
            guidance += "Frequency: Take 1-2 full-length mocks weekly in last 2-3 months. More frequent initially can hamper syllabus completion. Focus on quality analysis over quantity of mocks. Each mock should teach you something new about exam strategy or subject knowledge.\n\n"
            
            guidance += "Common Mistakes to Avoid:\n"
            guidance += "- Not taking mocks seriously\n"
            guidance += "- Skipping analysis after test\n"
            guidance += "- Getting demotivated by low scores\n"
            guidance += "- Not timing practice properly\n"
            guidance += "- Ignoring repeated mistakes\n\n"
            
            guidance += f"Remember: Mock tests are learning tools, not final judgments. Low scores indicate areas needing work, not your final {exam_name} result. Learn, improve, and move forward with each test.\n"
            
            self.dataset.append(guidance)
            self.line_count += guidance.count('\n')

    def generate_dataset(self):
        print("Starting dataset generation...")
        print(f"Target: {self.target_lines} lines")
        
        self.generate_educational_content(int(self.target_lines * 0.40))
        print(f"Progress: {self.line_count}/{self.target_lines} lines")
        
        self.generate_roadmaps(int(self.target_lines * 0.20))
        print(f"Progress: {self.line_count}/{self.target_lines} lines")
        
        self.generate_mental_health_conversations(int(self.target_lines * 0.20))
        print(f"Progress: {self.line_count}/{self.target_lines} lines")
        
        self.generate_exam_preparation_content(int(self.target_lines * 0.20))
        print(f"Progress: {self.line_count}/{self.target_lines} lines")
        
        print(f"\nDataset generation complete!")
        print(f"Total lines generated: {self.line_count}")
        print(f"Total conversations: {len(self.dataset)}")
        
        return self.dataset

    def save_dataset(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in self.dataset:
                f.write(entry + "\n")
        
        print(f"Dataset saved to: {output_path}")
        
        stats = {
            "total_lines": self.line_count,
            "total_conversations": len(self.dataset),
            "generated_at": datetime.now().isoformat(),
            "categories": {
                "educational": "40%",
                "roadmaps": "20%",
                "mental_health": "20%",
                "exam_preparation": "20%"
            }
        }
        
        stats_path = output_path.replace('.txt', '_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Statistics saved to: {stats_path}")

if __name__ == "__main__":
    generator = DatasetGenerator()
    dataset = generator.generate_dataset()
    generator.save_dataset("/workspaces/NextAI/data/processed/training_data.txt")
    
    print("\n✅ Dataset generation successful!")
    print(f"📊 Total lines: {generator.line_count}")
    print(f"💬 Total conversations: {len(dataset)}")
    print("\nDataset is ready for training!")