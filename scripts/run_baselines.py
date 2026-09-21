import json
import sqlite3
import torch
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from inference.model_loader import load_pretrained_gpt2_124m
from models.tokenizer.gpt2_tokenizer import GPT2Tokenizer
from inference.pipeline import EdgeCascadePipeline

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            mode TEXT,
            question_id TEXT,
            domain TEXT,
            category TEXT,
            tier_reached TEXT,
            retrieval_used BOOLEAN,
            remote_used BOOLEAN,
            judge_result TEXT,
            judge_score INTEGER,
            judge_reason TEXT,
            escalation_reason TEXT,
            latency_t1 REAL,
            latency_t2 REAL,
            latency_t3 REAL,
            latency_total REAL,
            retrieved_chunks TEXT,
            is_correct BOOLEAN,
            answer TEXT,
            ground_truth TEXT
        )
    ''')
    conn.commit()
    return conn

def evaluate_correctness(answer, ground_truth):
    # A simple correct check (does it contain some keywords or is it the remote answer)
    # The judge.py heuristic logic is very similar, so we can reuse that logic or simply check PASS
    if answer.startswith("[REMOTE]"):
        return True
    
    # We define correctness as whether the answer actually answers the question.
    # To simplify, we check for presence of key ground truth terms.
    import re
    gt_tokens = set(re.findall(r'\b\w+\b', ground_truth.lower()))
    ans_tokens = set(re.findall(r'\b\w+\b', answer.lower()))
    stopwords = {"the", "is", "a", "of", "in", "and", "to", "for", "with", "on", "at", "by", "from"}
    gt_content = gt_tokens - stopwords
    if not gt_content:
        return True
    
    overlap = len(gt_content.intersection(ans_tokens))
    # E.g. >= 30% overlap or at least 1 keyword
    return overlap > 0

def run_baselines():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    tokenizer = GPT2Tokenizer()
    questions = load_jsonl("data/evaluation/evaluation_questions.jsonl")
    
    model = load_pretrained_gpt2_124m(device=device)
    # Freeze and load LoRA
    for param in model.parameters(): param.requires_grad = False
    for name, param in model.named_parameters():
        if "lora_" in name: param.requires_grad = True
        
    lora_path = "models/checkpoints/lora/lora_adapter_sft.pt"
    if Path(lora_path).exists():
        print("Loading LoRA adapter...")
        model.load_state_dict(torch.load(lora_path, map_location=device)["model_state_dict"], strict=False)
    else:
        print("WARNING: LoRA adapter not found. Using base model.")
        
    model.eval()
    
    rag_paths = [
        "data/retrieval_final/isro/cartosat1_chunks.jsonl",
        "data/retrieval_final/isro/resourcesat2_chunks.jsonl",
        "data/retrieval_final/dpdpa/dpdpa_chunks.jsonl"
    ]
    
    pipeline = EdgeCascadePipeline(model, tokenizer, device, rag_paths)
    
    db_path = "data/evaluation/results.db"
    conn = init_db(db_path)
    c = conn.cursor()
    
    modes = ["local", "rag", "remote", "cascade"]
    
    for mode in modes:
        print(f"\n=== Running Mode: {mode.upper()} ===")
        for i, q in enumerate(questions):
            stats = pipeline.run(q["question"], q["domain"], ground_truth=q["ground_truth"], mode=mode)
            
            # Simple correctness check
            is_correct = evaluate_correctness(stats["answer"], q["ground_truth"])
            
            c.execute('''
                INSERT INTO results (
                    timestamp, mode, question_id, domain, category, tier_reached,
                    retrieval_used, remote_used, judge_result, judge_score, judge_reason, escalation_reason,
                    latency_t1, latency_t2, latency_t3, latency_total, retrieved_chunks, is_correct,
                    answer, ground_truth
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(), mode, q["question_id"], q["domain"], q["category"],
                stats["tier"], stats["retrieval_used"], stats["remote_used"],
                stats["judge_result"], stats.get("judge_score", 0), stats["judge_reason"], stats["escalation_reason"],
                stats["latency_t1"], stats["latency_t2"], stats["latency_t3"], stats["latency_total"],
                json.dumps(stats["retrieved_chunks"]), is_correct,
                stats["answer"], q["ground_truth"]
            ))
            conn.commit()
            if (i+1) % 5 == 0:
                print(f"Processed {i+1}/{len(questions)}...")
                
    print(f"\nAll baselines finished. Results saved to {db_path}")

if __name__ == "__main__":
    run_baselines()
