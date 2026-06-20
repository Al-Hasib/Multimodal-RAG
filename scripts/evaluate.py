#!/usr/bin/env python3
"""Evaluate RAG pipeline using ragas."""
import argparse
import json
import os
import sys
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
    context_recall,
    answer_correctness,
)


def load_test_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_evaluation(
    test_set_path: str,
    api_url: str = "http://localhost:8000",
    output_path: str | None = None,
):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.pipeline.pipeline import RAGPipeline

    pipe = RAGPipeline()
    test_set = load_test_set(test_set_path)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_set:
        import asyncio
        result = asyncio.run(pipe.aquery(item["question"]))
        questions.append(item["question"])
        answers.append(result.answer)
        contexts.append(result.context_texts)
        ground_truths.append([item.get("ground_truth", item["question"])])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    metrics = [
        faithfulness,
        answer_relevancy,
        context_relevancy,
        context_recall,
        answer_correctness,
    ]

    result = evaluate(dataset, metrics=metrics)
    print("\n=== Evaluation Results ===\n")
    print(result)

    if output_path:
        with open(output_path, "w") as f:
            f.write(result.to_json())
        print(f"\nResults saved to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("test_set", type=str, help="Path to test set JSON")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--output", type=str, help="Output path for results JSON")
    args = parser.parse_args()

    run_evaluation(args.test_set, args.api_url, args.output)


if __name__ == "__main__":
    main()
