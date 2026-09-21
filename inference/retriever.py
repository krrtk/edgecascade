import json
import math
import re
import numpy as np
from collections import defaultdict
from pathlib import Path

class SimpleTFIDFRetriever:
    def __init__(self, corpus_paths):
        self.documents = []
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []
        
        self.load_corpus(corpus_paths)
        self.build_index()

    def tokenize(self, text):
        # Simple lowercase word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words

    def load_corpus(self, paths):
        for path in paths:
            if not Path(path).exists():
                print(f"Warning: {path} not found.")
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    record = json.loads(line)
                    self.documents.append(record)

    def build_index(self):
        print(f"Building TF-IDF index for {len(self.documents)} chunks...")
        df = defaultdict(int)
        
        # Calculate Term Frequency (TF) for each doc and Document Frequency (DF)
        doc_tfs = []
        for doc in self.documents:
            tokens = self.tokenize(doc["text"])
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            
            # update df
            for token in set(tokens):
                df[token] += 1
                
            doc_tfs.append((len(tokens), tf))
            
        N = len(self.documents)
        # Assign vocab IDs and compute IDF
        for i, (token, count) in enumerate(df.items()):
            self.vocab[token] = i
            # Standard IDF: log(N / (1 + df)) + 1
            self.idf[token] = math.log(N / (1 + count)) + 1
            
        vocab_size = len(self.vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        # Build dense document vectors
        self.doc_vectors = np.zeros((N, vocab_size), dtype=np.float32)
        
        for doc_idx, (total_terms, tf) in enumerate(doc_tfs):
            if total_terms == 0:
                continue
            for token, count in tf.items():
                vocab_idx = self.vocab[token]
                # TF = count / total_terms
                val = (count / total_terms) * self.idf[token]
                self.doc_vectors[doc_idx, vocab_idx] = val
                
            # L2 normalize
            norm = np.linalg.norm(self.doc_vectors[doc_idx])
            if norm > 0:
                self.doc_vectors[doc_idx] /= norm

    def retrieve(self, query, top_k=3, domain=None):
        tokens = self.tokenize(query)
        if not tokens:
            return []
            
        q_vec = np.zeros(len(self.vocab), dtype=np.float32)
        total_terms = len(tokens)
        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
            
        for token, count in tf.items():
            if token in self.vocab:
                vocab_idx = self.vocab[token]
                val = (count / total_terms) * self.idf[token]
                q_vec[vocab_idx] = val
                
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
            
        # Cosine similarity
        sims = np.dot(self.doc_vectors, q_vec)
        
        results = []
        for i, sim in enumerate(sims):
            doc = self.documents[i]
            if domain and doc.get("domain") != domain:
                continue
            results.append((sim, doc))
            
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

# For testing
if __name__ == "__main__":
    paths = [
        "data/retrieval_final/isro/cartosat1_chunks.jsonl",
        "data/retrieval_final/isro/resourcesat2_chunks.jsonl",
        "data/retrieval_final/dpdpa/dpdpa_chunks.jsonl"
    ]
    retriever = SimpleTFIDFRetriever(paths)
    
    print("\nTest Retrieval 1:")
    res = retriever.retrieve("What is the swath of Cartosat-1 in stereo mode?", top_k=2, domain="isro")
    for sim, doc in res:
        print(f"Score: {sim:.4f} | Source: {doc['chunk_id']} | Text: {doc['text'][:50]}...")
        
    print("\nTest Retrieval 2:")
    res = retriever.retrieve("What are the conditions for consent under DPDPA?", top_k=2, domain="dpdpa")
    for sim, doc in res:
        print(f"Score: {sim:.4f} | Source: {doc['chunk_id']} | Text: {doc['text'][:50]}...")
