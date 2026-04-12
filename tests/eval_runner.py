"""
Eval Script — runs all 9 assignment questions + nice-to-haves through the router
and records responses for evaluation.

This tests the end-to-end pipeline: routing → query execution → response formatting.
Requires the FAISS index to be built (will build on first run).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocess import get_clean_dataframe
from src.semantic_queries import get_index_and_metadata
from src.router import route_query

# All assignment test questions
EVAL_QUESTIONS = [
    {
        "id": "Q1",
        "question": "When did The Matrix release?",
        "expected_type": "lookup",
        "expected_contains": ["1999"],
    },
    {
        "id": "Q2",
        "question": "What are the top 5 movies of 2019 by meta score?",
        "expected_type": "structured",
        "expected_contains": [],
    },
    {
        "id": "Q3",
        "question": "Top 7 comedy movies between 2010-2020 by IMDB rating?",
        "expected_type": "structured",
        "expected_contains": ["Comedy"],
    },
    {
        "id": "Q4",
        "question": "Top horror movies with a meta score above 85 and IMDB rating above 8",
        "expected_type": "structured",
        "expected_contains": ["Horror"],
    },
    {
        "id": "Q5",
        "question": "Top directors and their highest grossing movies with gross earnings of greater than 500M at least twice.",
        "expected_type": "director_aggregation",
        "expected_contains": [],
    },
    {
        "id": "Q6",
        "question": "Top 10 movies with over 1M votes but lower gross earnings.",
        "expected_type": "structured",
        "expected_contains": [],
    },
    {
        "id": "Q7",
        "question": "List of movies from the comedy genre where there is death or dead people involved.",
        "expected_type": "semantic",
        "expected_contains": ["Comedy"],
    },
    {
        "id": "Q8",
        "question": "Summarize the movie plots of Steven Spielberg's top-rated sci-fi movies.",
        "expected_type": "hybrid",
        "expected_contains": [],  # LLM summary may not repeat director name
    },
    {
        "id": "Q9",
        "question": "List of movies before 1990 that have involvement of police in the plot",
        "expected_type": "semantic",
        "expected_contains": [],
    },
    {
        "id": "NH1",
        "question": "List of Al Pacino movies that have grossed over $50M and IMDB rating of 8 or higher.",
        "expected_type": "clarification",  # Should ask lead vs any role per assignment
        "expected_contains": ["Al Pacino"],
    },
]


def run_eval():
    """Run all evaluation questions and record results."""
    print("=" * 70)
    print("IMDB Movie Assistant — End-to-End Evaluation")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading and preprocessing dataset...")
    df = get_clean_dataframe()

    print("[2/3] Building/loading FAISS index...")
    get_index_and_metadata(df)

    print("[3/3] Running evaluation questions...\n")
    print("-" * 70)

    results = []
    total_pass = 0
    total_fail = 0

    for eval_q in EVAL_QUESTIONS:
        qid = eval_q["id"]
        question = eval_q["question"]
        expected_type = eval_q["expected_type"]
        expected_contains = eval_q["expected_contains"]

        print(f"\n{'='*60}")
        print(f"[{qid}] {question}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            response_text, query_type, raw_result = route_query(question, df)
            elapsed = time.time() - start_time

            # Check routing correctness
            type_match = query_type == expected_type or (
                # Allow some flexibility in routing
                query_type in ("structured", "lookup") and expected_type in ("structured", "lookup")
            )

            # Check content
            content_match = True
            missing = []
            for expected in expected_contains:
                if expected.lower() not in response_text.lower():
                    content_match = False
                    missing.append(expected)

            # Has actual content
            has_content = len(response_text.strip()) > 50

            passed = type_match and content_match and has_content

            status = "✅ PASS" if passed else "❌ FAIL"
            if passed:
                total_pass += 1
            else:
                total_fail += 1

            print(f"\nStatus: {status}")
            print(f"Route type: {query_type} (expected: {expected_type}) {'✓' if type_match else '✗'}")
            print(f"Content check: {'✓' if content_match else '✗ missing: ' + str(missing)}")
            print(f"Has content: {'✓' if has_content else '✗'}")
            print(f"Time: {elapsed:.2f}s")
            print(f"\nResponse preview:\n{response_text[:500]}...")

            results.append({
                "id": qid,
                "question": question,
                "status": "PASS" if passed else "FAIL",
                "query_type": query_type,
                "expected_type": expected_type,
                "type_match": type_match,
                "content_match": content_match,
                "has_content": has_content,
                "elapsed_seconds": round(elapsed, 2),
                "response_preview": response_text[:1000],
                "full_response": response_text,
            })

        except Exception as e:
            elapsed = time.time() - start_time
            total_fail += 1
            print(f"\nStatus: ❌ ERROR")
            print(f"Error: {str(e)}")

            results.append({
                "id": qid,
                "question": question,
                "status": "ERROR",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
            })

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total: {len(EVAL_QUESTIONS)}")
    print(f"Passed: {total_pass}")
    print(f"Failed: {total_fail}")
    print(f"Pass rate: {total_pass / len(EVAL_QUESTIONS) * 100:.0f}%")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "..", "eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total": len(EVAL_QUESTIONS),
                "passed": total_pass,
                "failed": total_fail,
                "pass_rate": f"{total_pass / len(EVAL_QUESTIONS) * 100:.0f}%",
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    run_eval()
