from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import csv
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import (
    FEATURE_COLUMNS,
    LOGS_DIR,
    get_active_features,
    set_active_features,
)
from agents.alpha_agent.evaluate_alpha_model import evaluate
from agents.alpha_agent.feature_evaluator import evaluate_feature_importance
from agents.alpha_agent.train_alpha_model import train_alpha_model


MAX_ITERATIONS = 10
STABLE_RUNS = 5
NO_IMPROVEMENT_LIMIT = 3
TARGET_CORRELATION = 0.08
PROGRESS_PATH = LOGS_DIR / "auto_alpha_progress.csv"


def _stable_score_feature_set(features, cache):
    key = tuple(features)
    if key in cache:
        return cache[key]

    set_active_features(features, persist=False)
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        train_alpha_model(return_metrics=True)
        eval_metrics = evaluate(return_metrics=True)

    metrics = {
        "correlation": float(eval_metrics["correlation"]),
        "directional_accuracy": float(eval_metrics["directional_accuracy"]),
        "runs": STABLE_RUNS,
        "features": list(features),
    }
    cache[key] = metrics
    return metrics


def _choose_candidates(current_features, feature_stats):
    current_set = set(current_features)

    remove_candidate = None
    add_candidate = None

    active_stats = feature_stats[feature_stats["feature_name"].isin(current_set)]
    if len(current_features) > 1 and not active_stats.empty:
        remove_candidate = active_stats.sort_values(["signal_score", "stability_over_time"], ascending=[True, True]).iloc[0]["feature_name"]

    inactive_stats = feature_stats[~feature_stats["feature_name"].isin(current_set)]
    if not inactive_stats.empty:
        add_candidate = inactive_stats.sort_values(["signal_score", "stability_over_time"], ascending=[False, True]).iloc[0]["feature_name"]

    return remove_candidate, add_candidate


def _log_progress(rows):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "correlation", "directional_accuracy", "features"])
        writer.writeheader()
        writer.writerows(rows)


def run_auto_alpha(max_iterations=MAX_ITERATIONS):
    cache = {}
    progress_rows = []

    current_features = get_active_features()
    best_metrics = _stable_score_feature_set(current_features, cache)
    best_result = {
        "correlation": best_metrics["correlation"],
        "directional_accuracy": best_metrics["directional_accuracy"],
        "features": list(current_features),
    }

    no_improvement_count = 0
    last_trained_features = list(current_features)

    for iteration in range(1, max_iterations + 1):
        set_active_features(current_features, persist=False)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            feature_stats = evaluate_feature_importance()

        remove_candidate, add_candidate = _choose_candidates(current_features, feature_stats)
        candidate_feature_sets = []

        if remove_candidate is not None:
            candidate_feature_sets.append([feature for feature in current_features if feature != remove_candidate])
        if add_candidate is not None and add_candidate not in current_features:
            candidate_feature_sets.append(current_features + [add_candidate])

        candidate_feature_sets.append(list(current_features))

        evaluated_candidates = []
        for feature_set in candidate_feature_sets:
            if not feature_set:
                continue
            metrics = _stable_score_feature_set(feature_set, cache)
            evaluated_candidates.append(metrics)
            last_trained_features = list(feature_set)

        candidate_best = max(
            evaluated_candidates,
            key=lambda item: (item["correlation"], item["directional_accuracy"]),
        )

        if candidate_best["correlation"] > best_result["correlation"]:
            current_features = list(candidate_best["features"])
            best_result = {
                "correlation": candidate_best["correlation"],
                "directional_accuracy": candidate_best["directional_accuracy"],
                "features": list(candidate_best["features"]),
            }
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            current_features = list(best_result["features"])

        progress_rows.append(
            {
                "iteration": iteration,
                "correlation": best_result["correlation"],
                "directional_accuracy": best_result["directional_accuracy"],
                "features": "|".join(best_result["features"]),
            }
        )
        _log_progress(progress_rows)

        if best_result["correlation"] >= TARGET_CORRELATION:
            break
        if no_improvement_count >= NO_IMPROVEMENT_LIMIT:
            break

    if list(last_trained_features) != list(best_result["features"]):
        set_active_features(best_result["features"], persist=True)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            train_alpha_model(return_metrics=True)
    else:
        set_active_features(best_result["features"], persist=True)

    return best_result


if __name__ == "__main__":
    result = run_auto_alpha()
    print("===== AUTO ALPHA RESULT =====")
    print(f"Best Correlation: {result['correlation']:.6f}")
    print(f"Best Directional Accuracy: {result['directional_accuracy']:.6f}")
    print(f"Best Features: {result['features']}")
