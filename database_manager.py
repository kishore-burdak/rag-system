import json
import os

DB_FILE = "memory_stats.json"

def init_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump([], f)

def get_all_stats():
    init_db()
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def add_stat(filename, size_kb, num_chunks):
    stats = get_all_stats()
    # Check if already exists, update if it does
    for stat in stats:
        if stat["filename"] == filename:
            stat["size_kb"] = size_kb
            stat["num_chunks"] = num_chunks
            break
    else:
        stats.append({
            "filename": filename,
            "size_kb": size_kb,
            "num_chunks": num_chunks
        })
    
    with open(DB_FILE, "w") as f:
        json.dump(stats, f, indent=4)
