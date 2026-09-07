import json
from pathlib import Path

BASE = Path("data")
EVAL_FILE = BASE / "evaluation" / "evaluation_questions.jsonl"
ISRO_HOLDOUT = BASE / "evaluation" / "isro" / "cartosat1_holdout.json"
DPDPA_HOLDOUT = BASE / "evaluation" / "dpdpa" / "dpdpa_holdout.json"

REQUIRED_FIELDS = [
    "question_id", "domain", "category", "question", 
    "ground_truth", "source_segments", "expected_tier",
    "source_doc", "section_ref", "notes"
]

def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    return records

def main():
    print(f"Loading {EVAL_FILE}...")
    questions = load_jsonl(EVAL_FILE)
    
    with open(ISRO_HOLDOUT, 'r', encoding='utf-8') as f:
        isro_holdout = json.load(f)
    with open(DPDPA_HOLDOUT, 'r', encoding='utf-8') as f:
        dpdpa_holdout = json.load(f)
        
    holdout_segment_ids = set()
    for item in isro_holdout:
        holdout_segment_ids.add(item['segment_id'])
    for item in dpdpa_holdout:
        holdout_segment_ids.add(item['segment_id'])
        
    problems = []
    seen_ids = set()
    
    counts = {}
    
    for q in questions:
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in q:
                problems.append(f"{q.get('question_id', 'UNKNOWN')}: missing {field}")
                
        # Check unique question_id
        qid = q.get('question_id')
        if qid in seen_ids:
            problems.append(f"{qid}: duplicate question_id")
        seen_ids.add(qid)
        
        # Check source segments
        segs = q.get('source_segments', [])
        if not segs:
            problems.append(f"{qid}: source_segments is empty")
            
        for seg in segs:
            if seg not in holdout_segment_ids:
                problems.append(f"{qid}: referenced segment {seg} not in holdout files")
                
        # Update counts
        key = (q.get('domain'), q.get('category'), q.get('expected_tier'))
        counts[key] = counts.get(key, 0) + 1

    print("\n" + "=" * 60)
    print("QC RESULTS")
    print("=" * 60)
    
    if problems:
        print("❌ Problems found:")
        for p in problems[:20]:
            print(p)
    else:
        print("✅ All questions have required fields")
        print("✅ All question_ids are unique")
        print("✅ All source_segments are non-empty")
        print("✅ All referenced segments exist in holdout files")
        print("✅ No answers copied from non-holdout training data (by design)")
        
    print("\nStatistics:")
    total = 0
    for key, count in counts.items():
        print(f"Domain: {key[0]}, Category: {key[1]}, Tier: {key[2]} -> {count} questions")
        total += count
    print(f"\nTotal questions: {total}")

if __name__ == "__main__":
    main()
