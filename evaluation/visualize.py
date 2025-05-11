import json
import matplotlib.pyplot as plt
import pandas as pd


def extract_scores_from_json(json_data):
    correctness = [entry['answer_correctness'] for entry in json_data]
    faithfulness = [entry['faithfulness'] for entry in json_data]
    recall = [entry['context_recall'] for entry in json_data]

    return {
        "Answer Correctness": sum(correctness) / len(correctness),
        "Faithfulness": sum(faithfulness) / len(faithfulness),
        "Context Recall": sum(recall) / len(recall)
    }


def plot_evaluation_scores(json_input_path, output_chart_path):
    # Load from file
    with open(json_input_path, 'r') as f:
        data = json.load(f)

    # Extract scores
    summary = extract_scores_from_json(data)

    # Build DataFrame
    df = pd.DataFrame({
        "Metric": list(summary.keys()),
        "Average Score": list(summary.values())
    })

    # Plot
    plt.figure(figsize=(8, 4))
    bars = plt.bar(df["Metric"], df["Average Score"])
    plt.ylim(0, 1)
    plt.title("Athena RAG Evaluation Summary")
    plt.ylabel("Score (0 to 1)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Label bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig(output_chart_path)
    return output_chart_path

def main():
    json_input_path = "ragas_report.json"
    output_chart_path = "regas_results.png"

    result_path = plot_evaluation_scores(json_input_path, output_chart_path)

    if result_path is not None:
        print(f"Result successfully saved in {result_path}")

if __name__ == '__main__':
    main()

