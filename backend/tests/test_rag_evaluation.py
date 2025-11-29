"""
RAG System Evaluation
=====================

Production-grade metrics for RAG performance evaluation.

Measures:
- Retrieval Accuracy (relevance of retrieved documents)
- Query Response Time (P50, P95, P99)
- Cache Hit Rate (Redis performance)
- LLM Fallback Success Rate
- End-to-End System Performance

Run:
    python tests/test_rag_evaluation.py
"""

import sys
import time
from pathlib import Path
from statistics import mean, median
from typing import List, Dict, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag import rag_service
from app.services.cache import cache_service
from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service


# =============================================================================
# Test Queries (Diverse set for evaluation)
# =============================================================================

TEST_QUERIES = [
    # Specific technical skills queries (high accuracy expected)
    "What programming languages does Daniel know?",
    "What React and Next.js experience does Daniel have?",
    "What Python frameworks does Daniel use?",
    "What databases does Daniel work with?",
    "What blockchain technologies has Daniel worked with?",

    # Specific project queries (should retrieve well)
    "Tell me about AutoFlow Pro and its technology stack",
    "What is RataTutor and what was Daniel's role?",
    "Describe the NFT Trading Platform project",
    "What AI projects has Daniel built?",

    # Education and experience queries
    "Where did Daniel study computer science?",
    "What companies has Daniel worked for?",
    "What certifications does Daniel have?",
    "What was Daniel's GPA and academic performance?",

    # Mixed specificity (realistic queries)
    "Daniel's full-stack development experience",
    "Browser automation skills and tools",
    "Experience with distributed systems",

    # Vague queries (lower accuracy expected, but important to test)
    "skills",
    "projects",
    "background"
]


# =============================================================================
# Evaluation Metrics
# =============================================================================

class RAGEvaluator:
    """
    Comprehensive RAG system evaluation.

    Metrics:
    - Retrieval accuracy (relevance scoring)
    - Response time (latency)
    - Cache performance
    - System reliability
    """

    def __init__(self):
        self.results = {
            "retrieval_accuracy": [],
            "response_times": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_fallback_usage": {"ollama": 0, "groq": 0, "gemini": 0},
            "errors": []
        }

    def evaluate_retrieval_accuracy(self, query: str, sources: List[Dict]) -> float:
        """
        Evaluate retrieval accuracy using semantic similarity.

        Method:
        - Embed query using same model as RAG system
        - Embed each source document
        - Calculate cosine similarity
        - Average similarity across sources
        - Convert to 0-100% score

        Returns:
            Accuracy score 0-100% (higher = more relevant sources)
        """
        if not sources:
            return 0.0

        try:
            # Embed query
            query_embedding = embedding_service.embed_text(query)

            similarities = []
            for source in sources:
                # Get source content
                content = source.get('content_preview', '')
                if not content or content == '...':
                    continue

                # Embed source content
                source_embedding = embedding_service.embed_text(content)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, source_embedding)
                similarities.append(similarity)

            if not similarities:
                return 0.0

            # Average similarity across all sources
            avg_similarity = mean(similarities)

            # Convert to 0-100% (cosine similarity is -1 to 1, but typically 0 to 1 for text)
            # We normalize to 0-100% where:
            # - 0.0 similarity = 0%
            # - 1.0 similarity = 100%
            # - Typical good match is 0.6+ (60%+)
            accuracy = max(0, min(100, avg_similarity * 100))

            return accuracy

        except Exception as e:
            # Fallback to simple keyword matching on error
            print(f"[!] Semantic similarity error, using fallback: {e}")
            return self._keyword_fallback_accuracy(query, sources)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        # Cosine similarity = dot product / (norm1 * norm2)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _keyword_fallback_accuracy(self, query: str, sources: List[Dict]) -> float:
        """Simple keyword-based fallback evaluation"""
        query_lower = query.lower()
        keywords = [
            word for word in query_lower.split()
            if len(word) > 3 and word not in ['what', 'tell', 'about', 'does', 'have', 'with', 'that', 'this']
        ]

        if not keywords:
            return 0.0

        relevant_sources = 0
        for source in sources:
            content = source.get('content_preview', '').lower()
            # Check if any keyword appears in source
            if any(keyword in content for keyword in keywords):
                relevant_sources += 1

        return (relevant_sources / len(sources)) * 100 if sources else 0.0

    def measure_query_performance(
        self,
        query: str,
        use_hybrid: bool = True,
        use_reranking: bool = False
    ) -> Dict[str, Any]:
        """
        Measure performance of a single query.

        Returns:
            Dict with timing, accuracy, and metadata
        """
        print(f"\n[Testing] '{query}'")
        print("-" * 70)

        start_time = time.time()

        try:
            response = rag_service.query(
                query,
                k=3,
                use_hybrid_search=use_hybrid,
                use_reranking=use_reranking,
                include_scores=True  # Get actual retrieval scores!
            )

            end_time = time.time()
            response_time = end_time - start_time

            # Check if from cache
            is_cached = "(cached)" in response.model_used

            # Use actual retrieval scores instead of re-computing!
            # Scores are already normalized to 0-1 range by hybrid/reranking
            if response.retrieval_scores and len(response.retrieval_scores) > 0:
                scores = response.retrieval_scores

                # Use weighted average (top results matter more in RAG)
                # Weight: 60% top result, 30% 2nd, 10% 3rd
                weights = [0.6, 0.3, 0.1]
                weighted_sum = sum(
                    score * weight
                    for score, weight in zip(scores, weights[:len(scores)])
                )
                total_weight = sum(weights[:len(scores)])
                avg_score = weighted_sum / total_weight if total_weight > 0 else mean(scores)

                # Convert to percentage
                accuracy = avg_score * 100
            else:
                # Fallback: compute semantic similarity
                accuracy = self.evaluate_retrieval_accuracy(query, response.sources)

            # Determine which LLM was used
            llm_used = "ollama"
            if "groq" in response.model_used.lower():
                llm_used = "groq"
            elif "gemini" in response.model_used.lower():
                llm_used = "gemini"

            print(f"[OK] Response time: {response_time:.2f}s")
            print(f"[OK] Retrieval accuracy: {accuracy:.1f}%")
            print(f"[OK] Sources: {response.num_sources}")
            print(f"[OK] LLM: {llm_used} {'(cached)' if is_cached else ''}")

            return {
                "success": True,
                "response_time": response_time,
                "accuracy": accuracy,
                "num_sources": response.num_sources,
                "is_cached": is_cached,
                "llm_used": llm_used,
                "answer_length": len(response.answer)
            }

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time

            print(f"[ERROR] Query failed: {str(e)}")

            return {
                "success": False,
                "response_time": response_time,
                "error": str(e)
            }

    def run_evaluation(self, queries: List[str], iterations: int = 2):
        """
        Run comprehensive evaluation across test queries.

        Args:
            queries: List of test queries
            iterations: Number of times to run each query (for cache testing)
        """
        print("=" * 70)
        print("RAG System Evaluation")
        print("=" * 70)
        print(f"\nTest queries: {len(queries)}")
        print(f"Iterations per query: {iterations}")
        print(f"Total tests: {len(queries) * iterations}")

        # Clear cache to start fresh
        print("\n[i] Clearing cache for clean evaluation...")
        cache_service.clear()

        for iteration in range(iterations):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'='*70}")

            for query in queries:
                result = self.measure_query_performance(
                    query,
                    use_hybrid=True,  # Use hybrid search (vector + BM25) with proper scores
                    use_reranking=True  # Use reranking with normalized scores
                )

                if result["success"]:
                    self.results["retrieval_accuracy"].append(result["accuracy"])
                    self.results["response_times"].append(result["response_time"])

                    if result["is_cached"]:
                        self.results["cache_hits"] += 1
                    else:
                        self.results["cache_misses"] += 1

                    self.results["llm_fallback_usage"][result["llm_used"]] += 1
                else:
                    self.results["errors"].append(result["error"])

        # Get final cache stats
        cache_stats = cache_service.get_stats()

        return cache_stats

    def print_summary(self, cache_stats: Dict):
        """Print evaluation summary with metrics"""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        # Retrieval Accuracy
        if self.results["retrieval_accuracy"]:
            avg_accuracy = mean(self.results["retrieval_accuracy"])
            median_accuracy = median(self.results["retrieval_accuracy"])
            print(f"\nRetrieval Accuracy:")
            print(f"   Average:  {avg_accuracy:.1f}%")
            print(f"   Median:   {median_accuracy:.1f}%")
            print(f"   Min:      {min(self.results['retrieval_accuracy']):.1f}%")
            print(f"   Max:      {max(self.results['retrieval_accuracy']):.1f}%")

        # Response Time
        if self.results["response_times"]:
            response_times = sorted(self.results["response_times"])
            p50 = response_times[len(response_times) // 2]
            p95 = response_times[int(len(response_times) * 0.95)]
            p99 = response_times[int(len(response_times) * 0.99)]

            print(f"\nResponse Time:")
            print(f"   Average:  {mean(response_times):.2f}s")
            print(f"   Median:   {median(response_times):.2f}s")
            print(f"   P95:      {p95:.2f}s")
            print(f"   P99:      {p99:.2f}s")
            print(f"   Min:      {min(response_times):.2f}s")
            print(f"   Max:      {max(response_times):.2f}s")

        # Cache Performance
        total_queries = self.results["cache_hits"] + self.results["cache_misses"]
        cache_hit_rate = (self.results["cache_hits"] / total_queries * 100) if total_queries > 0 else 0

        print(f"\nCache Performance:")
        print(f"   Hit Rate:     {cache_hit_rate:.1f}%")
        print(f"   Hits:         {self.results['cache_hits']}")
        print(f"   Misses:       {self.results['cache_misses']}")
        print(f"   Cached Items: {cache_stats.get('cached_entries', 0)}")
        print(f"   Type:         {cache_stats.get('cache_type', 'unknown')}")
        print(f"   Redis:        {'Yes' if cache_stats.get('redis_connected') else 'No'}")

        # LLM Fallback Usage
        total_llm_calls = sum(self.results["llm_fallback_usage"].values())
        print(f"\nLLM Usage:")
        for llm, count in self.results["llm_fallback_usage"].items():
            percentage = (count / total_llm_calls * 100) if total_llm_calls > 0 else 0
            print(f"   {llm.capitalize():8} {count:3} calls ({percentage:.1f}%)")

        # System Reliability
        total_tests = len(self.results["retrieval_accuracy"]) + len(self.results["errors"])
        success_rate = (len(self.results["retrieval_accuracy"]) / total_tests * 100) if total_tests > 0 else 0

        print(f"\nSystem Reliability:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Errors:       {len(self.results['errors'])}")

        if self.results["errors"]:
            print(f"\nErrors encountered:")
            for i, error in enumerate(self.results["errors"][:5], 1):
                print(f"   {i}. {error}")

        # Vector Database Stats
        print(f"\nVector Database:")
        try:
            db_stats = vector_store.get_stats()
            print(f"   Documents:    {db_stats['total_documents']}")
            print(f"   Collection:   {db_stats['collection_name']}")
            print(f"   Embedding:    {db_stats['embedding_model']}")
            print(f"   Dimension:    {db_stats['embedding_dimension']}")
        except Exception as e:
            print(f"   Error getting stats: {e}")

        print("\n" + "=" * 70)
        print("PRODUCTION READINESS CHECKLIST")
        print("=" * 70)

        # Evaluation checklist
        checks = []

        # Retrieval accuracy >= 60% (hybrid + reranking target)
        # Note: With hybrid search + reranking, 60%+ is achievable
        # Basic vector search: 40-50%, Hybrid: 50-60%, Hybrid+Reranking: 60-80%
        if self.results["retrieval_accuracy"]:
            avg_acc = mean(self.results["retrieval_accuracy"])
            checks.append(("Retrieval Accuracy >=60%", avg_acc >= 60, f"{avg_acc:.1f}%"))

        # Response time P95 < 3s (excluding Ollama cold starts)
        if response_times:
            # Filter out first query (cold start)
            warm_times = response_times[1:] if len(response_times) > 1 else response_times
            if warm_times:
                warm_p95 = sorted(warm_times)[int(len(warm_times) * 0.95)] if len(warm_times) > 1 else warm_times[0]
                checks.append(("Response Time P95 <3s (warm)", warm_p95 < 3.0, f"{warm_p95:.2f}s"))
            else:
                checks.append(("Response Time P95 <3s", p95 < 3.0, f"{p95:.2f}s"))

        # Cache working
        checks.append(("Redis Cache Working", cache_stats.get('redis_connected', False), cache_stats.get('cache_type', 'none')))

        # System reliability >= 95%
        checks.append(("System Reliability >=95%", success_rate >= 95, f"{success_rate:.1f}%"))

        # Print checklist
        for check_name, passed, value in checks:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"{status} {check_name:30} {value}")

        print("=" * 70)


# =============================================================================
# Main Evaluation
# =============================================================================

if __name__ == "__main__":
    print("\nStarting RAG System Evaluation\n")

    # Force clear cache first to get fresh metrics with scores
    print("[i] Clearing cache to ensure fresh evaluation with retrieval scores...")
    cache_service.clear()
    print("")

    # Initialize evaluator
    evaluator = RAGEvaluator()

    # Run evaluation (1 iteration to get real retrieval scores)
    # Note: Running once ensures we get actual vector DB scores, not cached previews
    cache_stats = evaluator.run_evaluation(TEST_QUERIES, iterations=1)

    # Print summary
    evaluator.print_summary(cache_stats)

    print("\nEvaluation complete!\n")
