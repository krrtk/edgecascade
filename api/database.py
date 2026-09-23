import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join("data", "edgecascade.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            question TEXT,
            domain TEXT,
            tier TEXT,
            answer TEXT,
            latency REAL,
            rag_used BOOLEAN,
            remote_used BOOLEAN,
            feedback TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def log_request(req_id: str, question: str, domain: str, stats: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO requests (id, timestamp, question, domain, tier, answer, latency, rag_used, remote_used, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        req_id,
        timestamp,
        question,
        domain,
        stats.get("tier", "unknown"),
        stats.get("answer", ""),
        stats.get("latency_total", 0.0),
        stats.get("retrieval_used", False),
        stats.get("remote_used", False),
        None
    ))
    
    conn.commit()
    conn.close()

def update_feedback(req_id: str, feedback: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('UPDATE requests SET feedback = ? WHERE id = ?', (feedback, req_id))
    rows_affected = cursor.rowcount
    
    conn.commit()
    conn.close()
    
    return rows_affected > 0

def get_metrics() -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM requests')
    total = cursor.fetchone()[0]
    
    cursor.execute('SELECT tier, COUNT(*) FROM requests GROUP BY tier')
    tiers = {row[0]: row[1] for row in cursor.fetchall()}
    
    cursor.execute('SELECT COUNT(*) FROM requests WHERE rag_used = 1')
    rag_calls = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM requests WHERE remote_used = 1')
    remote_calls = cursor.fetchone()[0]
    
    cursor.execute('SELECT AVG(latency) FROM requests')
    avg_latency = cursor.fetchone()[0] or 0.0
    
    cursor.execute('SELECT feedback, COUNT(*) FROM requests WHERE feedback IS NOT NULL GROUP BY feedback')
    feedback_counts = {row[0]: row[1] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_requests": total,
        "tier_1_count": tiers.get("tier1", 0),
        "tier_2_count": tiers.get("tier2", 0),
        "tier_3_count": tiers.get("tier3", 0),
        "rag_calls": rag_calls,
        "remote_calls": remote_calls,
        "average_latency": round(avg_latency, 3),
        "positive_feedback": feedback_counts.get("positive", 0),
        "negative_feedback": feedback_counts.get("negative", 0)
    }
