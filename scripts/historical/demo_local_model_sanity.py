import json
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.practical_local_model import PracticalLocalModel

def run_sanity_test():
    model = PracticalLocalModel(device="cpu")
    
    test_questions = [
        # ISRO Questions
        {"domain": "ISRO", "question": "What is the full form of ISRO?"},
        {"domain": "ISRO", "question": "Who was the founder of the Indian Space Research Organisation?"},
        {"domain": "ISRO", "question": "What is the name of India's first satellite?"},
        {"domain": "ISRO", "question": "Which launch vehicle is known as the workhorse of ISRO?"},
        {"domain": "ISRO", "question": "What was the main objective of the Chandrayaan-3 mission?"},
        
        # DPDPA Questions
        {"domain": "DPDPA", "question": "What does DPDPA stand for in the context of Indian law?"},
        {"domain": "DPDPA", "question": "Who is considered a Data Fiduciary under the DPDPA?"},
        {"domain": "DPDPA", "question": "What is the role of the Data Protection Board of India?"},
        {"domain": "DPDPA", "question": "Are there penalties for non-compliance under the DPDPA?"},
        {"domain": "DPDPA", "question": "Does the DPDPA apply to personal data processed outside India?"}
    ]
    
    results = []
    print("\n--- Running Sanity Test ---")
    for item in test_questions:
        q = item["question"]
        domain = item["domain"]
        prompt = f"You are answering a question about {domain}. Please be concise and accurate.\n\n### Question:\n{q}\n\n### Answer:\n"
        
        start = time.time()
        ans = model.generate(prompt, max_new_tokens=50, return_probs=False)
        latency = time.time() - start
        
        print(f"\nDOMAIN: {domain}")
        print(f"QUESTION: {q}")
        print(f"LOCAL ANSWER: {ans}")
        print(f"LATENCY: {latency:.2f}s")
        
        results.append({
            "domain": domain,
            "question": q,
            "answer": ans,
            "latency": latency
        })
        
    with open("results/raw/practical_local_sanity.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print("\nSaved sanity test results to results/raw/practical_local_sanity.json")

if __name__ == "__main__":
    run_sanity_test()
