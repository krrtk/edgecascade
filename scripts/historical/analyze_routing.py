import json
import sqlite3
import pandas as pd
from inference.judge import get_judge

def main():
    # 1. Load questions
    q_dict = {}
    with open('data/evaluation/evaluation_questions.jsonl', 'r') as f:
        for line in f:
            d = json.loads(line)
            q_dict[d['question_id']] = d
            
    # 2. Load DB
    conn = sqlite3.connect('results/final/final_results.db')
    df = pd.read_sql_query("SELECT * FROM results", conn)
    conn.close()

    local_df = df[df['mode'] == 'local'].tail(30).reset_index(drop=True)
    rag_df = df[df['mode'] == 'rag'].tail(30).reset_index(drop=True)
    cascade_df = df[df['mode'] == 'cascade'].tail(30).reset_index(drop=True)

    judge = get_judge()
    
    print("Q_ID | T1_Score | T1_Quality | T2_Score | T2_Quality | Final_Tier | T1_Ans_Snippet")
    print("-" * 120)

    t1_counts = {1:0, 2:0, 3:0, 4:0, 5:0}
    t2_counts = {1:0, 2:0, 3:0, 4:0, 5:0}

    results = []

    for i in range(30):
        q_id = local_df.iloc[i]['question_id']
        question = q_dict[q_id]['question']
        
        t1_ans = local_df.iloc[i]['answer']
        t2_ans = rag_df.iloc[i]['answer']
        
        c_row = cascade_df.iloc[i]
        final_tier = c_row['tier_reached']
        
        # Re-judge T1
        j1 = judge.judge(question=question, answer=t1_ans)
        score1 = j1.get('score', 1)
        pass1 = j1.get('pass', False)
        t1_counts[score1] = t1_counts.get(score1, 0) + 1
        
        # Re-judge T2 (without context for simplicity of score analysis, or we could pass context)
        j2 = judge.judge(question=question, answer=t2_ans)
        score2 = j2.get('score', 1)
        pass2 = j2.get('pass', False)
        t2_counts[score2] = t2_counts.get(score2, 0) + 1
        
        ans_snippet = t1_ans[:40].replace('\n', ' ')
        
        print(f"{q_id} | {score1} ({'P' if pass1 else 'F'}) | ? | {score2} ({'P' if pass2 else 'F'}) | ? | {final_tier} | {ans_snippet}...")
        
        results.append({
            "q_id": q_id,
            "question": question,
            "t1_ans": t1_ans,
            "t1_score": score1,
            "t1_reason": j1.get('reason', ''),
            "t2_ans": t2_ans,
            "t2_score": score2,
            "t2_reason": j2.get('reason', '')
        })
        
    print(f"\nT1 score distribution: {t1_counts}")
    print(f"T2 score distribution: {t2_counts}")
    
    with open("results/final/rejudge_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
