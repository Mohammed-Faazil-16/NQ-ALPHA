from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import statistics
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.evaluate_alpha_model import evaluate
from agents.alpha_agent.train_alpha_model import train_alpha_model


DEFAULT_RUNS = 5


def run_stable_alpha(runs=DEFAULT_RUNS):
    results = []

    for _ in range(runs):
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            train_alpha_model(return_metrics=True)
            metrics = evaluate(return_metrics=True)
        results.append(metrics)

    correlations = [result["correlation"] for result in results]
    directional_accuracies = [result["directional_accuracy"] for result in results]

    avg_correlation = statistics.fmean(correlations)
    std_correlation = statistics.pstdev(correlations) if len(correlations) > 1 else 0.0
    avg_directional_accuracy = statistics.fmean(directional_accuracies)

    print("===== FINAL STABLE RESULTS =====")
    print(f"Runs: {runs}")
    print(f"Average Correlation: {avg_correlation:.6f}")
    print(f"Correlation Std: {std_correlation:.6f}")
    print(f"Average Directional Accuracy: {avg_directional_accuracy:.6f}")


if __name__ == "__main__":
    run_stable_alpha()
