import sqlite3
import pandas as pd
import json
import os
import requests
import time

def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    os.environ[k] = v

load_env()

def audit_answer(question, answer, ground_truth):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("No GROQ_API_KEY for audit")
        return "UNKNOWN"
        
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    model = "qwen/qwen3.8-27b"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "You are an expert answer auditor. Evaluate the generated answer against the question and the ground truth. "
        "Classify the generated answer into exactly one of these categories:\n"
        "- GOOD: Sensible, coherent, directly answers the question, and factually correct.\n"
        "- PARTIALLY_GOOD: Understandable but missing some details, partially correct, or contains minor errors.\n"
        "- BAD: Nonsensical, hallucinated, highly repetitive, contradictory, or completely fails to answer the question.\n\n"
        "You MUST output JSON with this format:\n"
        "{\"quality\": \"GOOD\" | \"PARTIALLY_GOOD\" | \"BAD\", \"reason\": \"...\"}"
    )
    
    user_prompt = f"Question: {question}\nGround Truth: {ground_truth}\nGenerated Answer: {answer}"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    for attempt in range(3):
        try:
            r = requests.post(endpoint, headers=headers, json=payload, timeout=10)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            res = json.loads(content)
            return res.get("quality", "UNKNOWN")
        except Exception as e:
            if attempt == 2:
                print(f"Audit API Error: {e}")
                return "UNKNOWN"
            time.sleep(1)

def main():
    db_path = "data/evaluation/results.db"
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results WHERE mode='cascade' ORDER BY id DESC LIMIT 30", conn)
    conn.close()
    
    df = df.iloc[::-1].reset_index(drop=True)
    
    results = []
    counts = {"GOOD": 0, "PARTIALLY_GOOD": 0, "BAD": 0, "UNKNOWN": 0}
    
    # Load questions to get the question text
    q_dict = {}
    with open("data/evaluation/evaluation_questions.jsonl", "r") as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                q_dict[q["question_id"]] = q["question"]
    
    for idx, row in df.iterrows():
        question = q_dict.get(row["question_id"], row["question_id"])
        quality = audit_answer(question, row["answer"], row["ground_truth"])
        counts[quality] = counts.get(quality, 0) + 1
        
        results.append({
            "question_id": row["question_id"],
            "quality": quality,
            "answer": row["answer"]
        })
        print(f"Audited {idx+1}/30: {quality}")
        
    audit_summary = {
        "counts": counts,
        "percentages": {k: round(v / len(df) * 100, 1) for k, v in counts.items()},
        "details": results
    }
    
    os.makedirs("results/final", exist_ok=True)
    with open("results/final/answer_quality_audit.json", "w") as f:
        json.dump(audit_summary, f, indent=2)
        
    print("\nAudit Complete.")
    print(audit_summary["percentages"])

if __name__ == "__main__":
    main()
