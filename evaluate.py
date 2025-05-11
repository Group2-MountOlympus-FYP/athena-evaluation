import argparse
import json
import pandas as pd
from datasets import Dataset
import requests
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness, context_recall


def load_dataset(path: str) -> Dataset:
    """Read CSV or JSON with cols `question` + `ground_truth`."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path)
    return Dataset.from_pandas(df[["question", "ground_truth"]])


def annotate_with_athena(ds: Dataset, base_url: str) -> Dataset:
    """Run Athena via Flask API -> add `answer` and `contexts` columns that Ragas expects."""
    answers = []
    contexts = []
    for row in ds:
        # Call the generate endpoint for the answer
        resp = requests.post(f"{base_url.rstrip('/')}/generate", json={"query": row["question"]})
        resp.raise_for_status()
        data = resp.json()
        # Extract answer text
        answer = data["result"].get("result")
        answers.append(answer)

        # Call retrieve_documents_only endpoint for contexts
        resp_docs = requests.post(f"{base_url.rstrip('/')}/retrieve_documents_only", json={"query": row["question"]})
        resp_docs.raise_for_status()
        docs_data = resp_docs.json()
        docs = docs_data.get("documents", [])
        contexts.append(docs)

    ds = ds.add_column("answer", answers)
    ds = ds.add_column("contexts", contexts)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV/JSON with eval rows")
    ap.add_argument("--base-url", default="http://localhost:5000", help="Base URL for the Athena Flask API")
    ap.add_argument("--out", default="ragas_report.json",
                    help="Where to write the metrics (JSON)")
    args = ap.parse_args()

    # 1. Load evaluation questions
    eval_ds = load_dataset(args.dataset)

    # 2. Use Flask API base URL
    base_url = args.base_url

    # 3. Get answers + retrieved chunks via Flask API
    eval_ds = annotate_with_athena(eval_ds, base_url)

    # 4. Run Ragas
    report = evaluate(
        eval_ds,
        metrics=[answer_correctness, faithfulness, context_recall],
    )

    results = report.to_pandas()
    data = json.loads(results.to_json(orient="records"))

    # 5. Persist & print
    with open(args.out, "w") as fp:
        json.dump(data, fp, indent=2)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
