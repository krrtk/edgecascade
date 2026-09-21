import sqlite3
import pandas as pd
from pathlib import Path
import json

def generate_report():
    db_path = "data/evaluation/results.db"
    if not Path(db_path).exists():
        print("Database not found. Run baselines first.")
        return
        
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()
    
    print("\n" + "="*50)
    print("EDGE CASCADE - PHASE 3 FINAL BENCHMARK AUDIT")
    print("="*50 + "\n")
    
    modes = ["local", "rag", "remote", "cascade"]
    
    for mode in modes:
        mode_df = df[df['mode'] == mode]
        if len(mode_df) == 0:
            continue
            
        print(f"--- SYSTEM: {mode.upper()} ---")
        overall = mode_df['is_correct'].mean() * 100
        isro = mode_df[mode_df['domain'] == 'isro']['is_correct'].mean() * 100
        dpdpa = mode_df[mode_df['domain'] == 'dpdpa']['is_correct'].mean() * 100
        kg = mode_df[mode_df['category'] == 'knowledge_gap']['is_correct'].mean() * 100
        reasoning = mode_df[mode_df['category'] == 'reasoning']['is_correct'].mean() * 100
        avg_latency = mode_df['latency_total'].mean()
        
        rag_calls = mode_df['retrieval_used'].sum()
        remote_calls = mode_df['remote_used'].sum()
        remote_rate = (remote_calls / len(mode_df)) * 100
        
        print(f"Overall correctness: {overall:.1f}%")
        print(f"ISRO correctness: {isro:.1f}%")
        print(f"DPDPA correctness: {dpdpa:.1f}%")
        print(f"Knowledge-gap correctness: {kg:.1f}%")
        print(f"Reasoning correctness: {reasoning:.1f}%")
        if mode == "remote":
            print(f"Average latency: {avg_latency:.2f}s (Mocked/Simulated)")
        else:
            print(f"Average latency: {avg_latency:.2f}s")
            
        print(f"RAG calls: {rag_calls}")
        if mode == "remote":
            print(f"Remote calls: {remote_calls} ({remote_rate:.1f}%) - [MOCK REMOTE LLM]")
        else:
            print(f"Remote calls: {remote_calls} ({remote_rate:.1f}%)")
        print()
        
    # Escalation rates for cascade
    cascade_df = df[df['mode'] == 'cascade']
    if len(cascade_df) > 0:
        print("--- CASCADE ROUTING STATISTICS ---")
        tier1_final = len(cascade_df[cascade_df['tier_reached'] == 'tier1'])
        
        local_rag = len(cascade_df[(cascade_df['tier_reached'] == 'tier2') | 
                                   (cascade_df['tier_reached'] == 'tier3') & (cascade_df['retrieval_used'] == True)])
        
        # Local -> Remote (meaning Overlap gate failed)
        local_remote = len(cascade_df[(cascade_df['tier_reached'] == 'tier3') & (cascade_df['retrieval_used'] == False)])
        
        # RAG -> Remote (Tier 2 Judge Fail)
        rag_remote = len(cascade_df[(cascade_df['tier_reached'] == 'tier3') & (cascade_df['retrieval_used'] == True)])
        
        tier2_final = len(cascade_df[cascade_df['tier_reached'] == 'tier2'])
        tier3_final = len(cascade_df[cascade_df['tier_reached'] == 'tier3'])
        
        print(f"Tier 1 final answers: {tier1_final}")
        print(f"Tier 1 -> Tier 2 (Local -> RAG escalations): {local_rag}")
        print(f"Tier 1 -> Tier 3 (Local -> Remote Direct escalations): {local_remote}")
        print(f"Tier 2 -> Tier 3 (RAG -> Remote escalations): {rag_remote}")
        print(f"Final answers by tier: Tier 1 ({tier1_final}), Tier 2 ({tier2_final}), Tier 3 ({tier3_final})")
        print()
        
        # Failure analysis
        print("--- INDEPENDENT CORRECTNESS FAILURE ANALYSIS ---")
        failures = cascade_df[cascade_df['is_correct'] == False]
        print(f"Total cascade failures: {len(failures)}")
        
        if len(failures) > 0:
            print("\nDetailed Failures:")
            for _, row in failures.iterrows():
                # Categorize failure
                failure_type = "other"
                if row['tier_reached'] == 'tier1' or row['tier_reached'] == 'tier2':
                    failure_type = "judge_routing_failure" # Judge passed a wrong answer
                elif row['tier_reached'] == 'tier3':
                    failure_type = "remote_mock_failure"
                    
                print(f"Q ID: {row['question_id']} | Category: {row['category']}")
                print(f"Expected Tier: Not currently stored | Actual Final Tier: {row['tier_reached']}")
                print(f"Failure Type: {failure_type}")
                print(f"Generated Answer: {row['answer'][:150]}...")
                print(f"Ground Truth: {row['ground_truth'][:150]}...")
                print("-" * 30)

    # RAG Verification
    print("\n--- RAG RETRIEVAL VERIFICATION ---")
    rag_df = df[df['mode'] == 'rag']
    if len(rag_df) > 0:
        print("Sample of Retrieved Contexts for Knowledge-Gap Queries:")
        sample = rag_df[rag_df['category'] == 'knowledge_gap'].head(3)
        for _, row in sample.iterrows():
            chunks = json.loads(row['retrieved_chunks'])
            chunk_id = chunks[0] if chunks else "None"
            print(f"Q: {row['question_id']}")
            print(f"Retrieved Chunk ID: {chunk_id}")
            print(f"Is answer correct based on this context?: {row['is_correct']}")
            print("-" * 30)

if __name__ == "__main__":
    generate_report()
