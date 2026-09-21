import sqlite3
import pandas as pd
import json
import os
import shutil

def categorize_failure(row):
    if row['tier_reached'] in ('tier1', 'tier2'):
        return 'judge_routing_failure'
    elif row['tier_reached'] == 'tier3':
        return 'remote_fallback_failure' # Or remote_mock_failure
    return 'other'

def main():
    db_path = "data/evaluation/results.db"
    
    # 9. STRICT RESULT ARTIFACT PATHS
    os.makedirs("results/final", exist_ok=True)
    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    
    shutil.copy(db_path, "results/final/final_results.db")
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    
    # Only keep the most recent 30 entries for each mode
    # Assuming run_baselines.py inserts local, rag, remote, cascade sequentially
    df = df.groupby('mode').tail(30).reset_index(drop=True)
    
    # Export raw results
    for mode in ["local", "rag", "remote", "cascade"]:
        mode_df = df[df['mode'] == mode]
        if mode == "cascade":
            mode_df.to_json("results/raw/edgecascade_results.json", orient="records", indent=2)
        else:
            with open("results/raw/baseline_results.json", "a", encoding="utf-8") as f:
                mode_df.to_json(f, orient="records", indent=2, lines=True)

    metrics_list = []
    
    for mode in ["local", "rag", "remote", "cascade"]:
        mode_df = df[df['mode'] == mode]
        if len(mode_df) == 0: continue
        
        overall = mode_df['is_correct'].mean() * 100
        isro = mode_df[mode_df['domain'] == 'isro']['is_correct'].mean() * 100
        dpdpa = mode_df[mode_df['domain'] == 'dpdpa']['is_correct'].mean() * 100
        kg = mode_df[mode_df['category'] == 'knowledge_gap']['is_correct'].mean() * 100
        reasoning = mode_df[mode_df['category'] == 'reasoning']['is_correct'].mean() * 100
        avg_latency = mode_df['latency_total'].mean()
        
        rag_calls = int(mode_df['retrieval_used'].sum())
        remote_calls = int(mode_df['remote_used'].sum())
        remote_rate = (remote_calls / len(mode_df)) * 100
        
        metrics_list.append({
            "System": mode.capitalize(),
            "Overall correctness": round(overall, 1),
            "ISRO correctness": round(isro, 1),
            "DPDPA correctness": round(dpdpa, 1),
            "Knowledge-gap correctness": round(kg, 1),
            "Reasoning correctness": round(reasoning, 1),
            "Average latency": round(avg_latency, 2),
            "RAG calls": rag_calls,
            "Remote calls": remote_calls,
            "Remote-call rate": round(remote_rate, 1)
        })

    # Save metrics
    with open("results/final/final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_list, f, indent=2)
        
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv("results/final/final_metrics.csv", index=False)
    
    # Routing statistics
    cascade_df = df[df['mode'] == 'cascade']
    
    tier1_final = int((cascade_df['tier_reached'] == 'tier1').sum())
    tier1_to_tier2 = int(((cascade_df['tier_reached'] != 'tier1') & (cascade_df['retrieval_used'] == True)).sum())
    tier1_to_tier3 = int(((cascade_df['tier_reached'] != 'tier1') & (cascade_df['retrieval_used'] == False)).sum())
    tier2_final = int((cascade_df['tier_reached'] == 'tier2').sum())
    tier2_to_tier3 = int(((cascade_df['tier_reached'] == 'tier3') & (cascade_df['retrieval_used'] == True)).sum())
    tier3_final = int((cascade_df['tier_reached'] == 'tier3').sum())
    
    routing_stats = {
        "Tier 1 final": tier1_final,
        "Tier 1 -> Tier 2": tier1_to_tier2,
        "Tier 1 -> Tier 3": tier1_to_tier3,
        "Tier 2 final": tier2_final,
        "Tier 2 -> Tier 3": tier2_to_tier3,
        "Tier 3 final": tier3_final
    }
    
    with open("results/final/routing_summary.json", "w", encoding="utf-8") as f:
        json.dump(routing_stats, f, indent=2)
        
    # Failure analysis
    failures = cascade_df[cascade_df['is_correct'] == False]
    failure_list = []
    for _, row in failures.iterrows():
        failure_list.append({
            "question_id": row['question_id'],
            "domain": row['domain'],
            "category": row['category'],
            "question": row['question_id'], # We don't have the original question text in DB, but ID works
            "generated_answer": row['answer'],
            "ground_truth": row['ground_truth'],
            "final_tier": row['tier_reached'],
            "routing_decision": "PASS" if row['tier_reached'] in ('tier1', 'tier2') else "ESCALATE",
            "failure_type": categorize_failure(row)
        })
        
    with open("results/final/failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(failure_list, f, indent=2)
        
    # Representative examples
    examples = []
    # Successful Tier 1
    t1_success = cascade_df[(cascade_df['tier_reached'] == 'tier1') & (cascade_df['is_correct'] == True)].head(1)
    if not t1_success.empty:
        row = t1_success.iloc[0]
        examples.append({
            "type": "Successful Tier 1",
            "question_id": row['question_id'],
            "answer": row['answer'],
            "why_sufficient": "Answer was generated confidently with high average log-prob."
        })
        
    # Successful Tier 2
    t2_success = cascade_df[(cascade_df['tier_reached'] == 'tier2') & (cascade_df['is_correct'] == True)].head(1)
    if not t2_success.empty:
        row = t2_success.iloc[0]
        examples.append({
            "type": "Successful Tier 2",
            "question_id": row['question_id'],
            "retrieved_evidence": row['retrieved_chunks'],
            "answer": row['answer'],
            "why_helped": "Retrieval provided factual entities required for semantic routing judge approval."
        })
        
    # Successful Tier 3
    t3_success = cascade_df[(cascade_df['tier_reached'] == 'tier3') & (cascade_df['is_correct'] == True)].head(1)
    if not t3_success.empty:
        row = t3_success.iloc[0]
        examples.append({
            "type": "Successful Tier 3",
            "question_id": row['question_id'],
            "reason_t1_t2_failed": row['escalation_reason'],
            "remote_fallback_result": row['answer']
        })
        
    # Failure
    if len(failure_list) > 0:
        row = failure_list[0]
        examples.append({
            "type": "Failure",
            "question_id": row['question_id'],
            "incorrect_answer": row['generated_answer'],
            "routing_decision": row['routing_decision'],
            "why_failed": row['failure_type']
        })
        
    with open("results/final/representative_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
        
    print("Export complete.")

if __name__ == "__main__":
    main()
