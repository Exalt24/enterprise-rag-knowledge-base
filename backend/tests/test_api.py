"""
FastAPI Test Script

Tests all API endpoints using the requests library.

Make sure the API is running first:
    python -m app.main
    Or: uvicorn app.main:app --reload

Then run this script in another terminal.
"""

import requests
import json
from pathlib import Path


BASE_URL = "http://localhost:8001"


def test_health():
    """Test /health endpoint"""
    print("\n" + "=" * 70)
    print("TEST: Health Check")
    print("=" * 70)

    response = requests.get(f"{BASE_URL}/api/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_stats():
    """Test /stats endpoint"""
    print("\n" + "=" * 70)
    print("TEST: Database Stats")
    print("=" * 70)

    response = requests.get(f"{BASE_URL}/api/stats")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_query():
    """Test /query endpoint"""
    print("\n" + "=" * 70)
    print("TEST: Query Endpoint")
    print("=" * 70)

    test_questions = [
        "What is RAG?",
        "What embedding model is used?",
    ]

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("-" * 70)

        payload = {
            "question": question,
            "k": 3,
            "include_sources": True
        }

        response = requests.post(
            f"{BASE_URL}/api/query",
            json=payload
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nAnswer: {data['answer']}")
            print(f"\nSources ({data['num_sources']}):")
            for i, source in enumerate(data['sources'][:2], 1):  # Show first 2
                print(f"  [{i}] {source['file_name']}")
                if source.get('relevance_score'):
                    print(f"      Relevance: {source['relevance_score']:.4f}")
        else:
            print(f"Error: {response.text}")

    return True


def test_ingest():
    """Test /ingest endpoint"""
    print("\n" + "=" * 70)
    print("TEST: Ingest Endpoint")
    print("=" * 70)

    # Create test file
    test_file = Path("data/api_test.txt")
    test_file.write_text("""
This is a test document for the RAG API.

The API allows you to:
- Upload documents via POST /ingest
- Query the knowledge base via POST /query
- Check system health via GET /health
- View statistics via GET /stats

All endpoints are documented at /docs with interactive testing.
    """)

    print(f"Uploading: {test_file.name}")
    print("-" * 70)

    # Upload file
    with open(test_file, "rb") as f:
        files = {"file": (test_file.name, f, "text/plain")}
        response = requests.post(f"{BASE_URL}/api/ingest", files=files)

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    # Cleanup
    test_file.unlink()

    return response.status_code == 200


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FASTAPI ENDPOINT TESTS")
    print("=" * 70)
    print("\nMake sure the API is running:")
    print("  Terminal 1: python -m app.main")
    print("  Terminal 2: python test_api.py")
    print("=" * 70)

    try:
        # Test endpoints
        results = {
            "health": test_health(),
            "stats": test_stats(),
            "query": test_query(),
            "ingest": test_ingest(),
        }

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        for endpoint, passed in results.items():
            status = "[OK]" if passed else "[FAIL]"
            print(f"{status} {endpoint}")

        all_passed = all(results.values())

        if all_passed:
            print("\n" + "=" * 70)
            print("[OK] ALL TESTS PASSED!")
            print("=" * 70)
            print("\nYour RAG API is fully functional!")
            print("\nNext steps:")
            print("  1. Visit http://localhost:8000/docs for interactive docs")
            print("  2. Try uploading your own documents")
            print("  3. Ask questions via the API")
        else:
            print("\n[X] Some tests failed")

    except requests.exceptions.ConnectionError:
        print("\n[X] Could not connect to API!")
        print("\nMake sure the API is running:")
        print("  python -m app.main")
        print("  Or: uvicorn app.main:app --reload")
