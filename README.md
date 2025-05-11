# Athena Evaluation

This repository contains code for evaluating Athena Intelligence, the RAG system for **Prometheus.EDU**.

## Evaluation Framework

We leverage **Ragas** as our evaluation framework.

## How to run?

To run this project you should first have a `.env` file containing the following environmental variables.

```text
GOOGLE_API_KEY=""
OPENAI_API_KEY=""
```

Then run the following commands to evaluate the **Athena Intelligence** RAG system.

```shell
uv run evaluation.py --dataset eval.csv --out ragas_report.json # for evaluation
# and
uv run visualize.py # for visualizing the results
```

> [!NOTE]
> 1. `uv` package manager is required for installing the dependecies.
> 2. You should have the Flask server running to evaluate it.
> 3. An additional flag `--base-url` can be added if your Flask server is not running on the default port.
