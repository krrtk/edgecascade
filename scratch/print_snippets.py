import json
from pathlib import Path

def print_snippets(path):
    print(f"=== {path} ===")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for i, item in enumerate(data):
        print(f"ID: {item['segment_id']}")
        print(item['text'].strip())
        print("-" * 40)
        if i >= 10: break

print_snippets('data/evaluation/isro/cartosat1_holdout.json')
print_snippets('data/evaluation/dpdpa/dpdpa_holdout.json')
