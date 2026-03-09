"""
Cloud Services Keep-Alive

Pings free-tier cloud services to prevent inactivity deletion:
- Qdrant Cloud: Suspends after 1 week, deletes after 4 weeks
- Redis Cloud: Deletes after 30 days

Run at least every 5 days to keep both alive.

Usage:
    python keep_alive.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

failures = 0

# --- Qdrant Cloud ---
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

if qdrant_url and qdrant_key:
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10.0)
        collections = client.get_collections()
        print(f"[OK] Qdrant alive - {len(collections.collections)} collection(s)")
    except Exception as e:
        print(f"[ERROR] Qdrant ping failed: {e}")
        failures += 1
else:
    print("[SKIP] Qdrant - no credentials set")

# --- Redis Cloud ---
redis_url = os.getenv("REDIS_URL")

if redis_url:
    try:
        import redis
        r = redis.from_url(redis_url, socket_timeout=10)
        r.ping()
        print(f"[OK] Redis alive - {r.dbsize()} key(s)")
    except Exception as e:
        print(f"[ERROR] Redis ping failed: {e}")
        failures += 1
else:
    print("[SKIP] Redis - no URL set")

if failures:
    sys.exit(1)
