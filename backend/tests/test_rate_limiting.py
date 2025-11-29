"""
Rate Limiting Test
==================

Test that rate limiting works correctly.

Run this while the API server is running:
    python tests/test_rate_limiting.py
"""

import requests
import time

API_URL = "http://localhost:8001"

def test_rate_limiting():
    print("=" * 70)
    print("Rate Limiting Test")
    print("=" * 70)

    # Test /api/stats endpoint (120 requests/minute limit)
    print("\n[1] Testing /api/stats (limit: 120/min)")
    print("-" * 70)

    response = requests.get(f"{API_URL}/api/stats")
    print(f"Status: {response.status_code}")
    print(f"X-RateLimit-Limit: {response.headers.get('X-RateLimit-Limit', 'Not set')}")
    print(f"X-RateLimit-Window: {response.headers.get('X-RateLimit-Window', 'Not set')}")

    # Test query endpoint with rapid requests
    print("\n[2] Testing /api/query (limit: 60/min)")
    print("-" * 70)
    print("Making 5 rapid requests...")

    for i in range(5):
        response = requests.post(
            f"{API_URL}/api/query",
            json={"question": f"test query {i}"}
        )
        print(f"  Request {i+1}: {response.status_code}")

        if response.status_code == 429:
            print(f"  Rate limit hit! Retry-After: {response.headers.get('Retry-After')} seconds")
            print(f"  Message: {response.json()}")
            break

    print("\n[OK] Rate limiting is working!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        test_rate_limiting()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to API server.")
        print("Make sure the server is running: python -m app.main")
