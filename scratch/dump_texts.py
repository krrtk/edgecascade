import json

def dump_texts(path, outpath):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open(outpath, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data):
            f.write(f"ID: {item['segment_id']}\n")
            f.write(item['text'].strip() + "\n")
            f.write("-" * 40 + "\n")

dump_texts('data/evaluation/isro/cartosat1_holdout.json', 'scratch/carto_dump.txt')
dump_texts('data/evaluation/dpdpa/dpdpa_holdout.json', 'scratch/dpdpa_dump.txt')
