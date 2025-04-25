"""
Evaluate the Athena RAG pipeline with Ragas.

Usage:
    python evaluation.py --dataset eval.csv --out ragas_report.json
"""

import argparse, json
import pandas as pd
from datasets import Dataset
from athena_ta_core import create_athena_client
from ragas import evaluate
from ragas.metrics import answer_correctness, faithfulness, context_recall


def load_dataset(path: str) -> Dataset:
    """Read CSV or JSON with cols `question` + `ground_truth`."""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_json(path)
    return Dataset.from_pandas(df[["question", "ground_truth"]])

def annotate_with_athena(ds: Dataset, athena) -> Dataset:
    """Run Athena → add `answer` and `contexts` columns that Ragas expects."""
    answers, contexts = [], []
    for row in ds:
        res = athena.generate(row["question"])          # RetrievalQA chain
        answers.append(res["result"])
        # NB: your RetrievalQA must be built with `return_source_documents=True`
        ctx = [d.page_content for d in res["source_documents"]]
        contexts.append(ctx)
    ds = ds.add_column("answer", answers)
    ds = ds.add_column("contexts", contexts)
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV/JSON with eval rows")
    ap.add_argument("--out", default="ragas_report.json",
                    help="Where to write the metrics (JSON)")
    args = ap.parse_args()

    # 1. Load evaluation questions
    eval_ds = load_dataset(args.dataset)

    # 2. Spin up Athena once
    athena = create_athena_client()

    # 3. Get answers + retrieved chunks
    eval_ds = annotate_with_athena(eval_ds, athena)

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
