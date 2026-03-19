from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import (
    LOGS_DIR,
    build_modeling_dataframes,
    get_active_features,
    reset_active_features,
    set_active_features,
)
from agents.alpha_agent.evaluate_alpha_model import evaluate
from agents.alpha_agent.feature_evaluator import evaluate_feature_importance, filter_features, select_top_features
from agents.alpha_agent.train_alpha_model import train_alpha_model


RESEARCH_RESULTS_PATH = LOGS_DIR / "alpha_research_results.csv"
ABLATION_RESULTS_PATH = LOGS_DIR / "ablation_results.csv"
ABLATION_MODEL_DIR = PROJECT_ROOT / "models" / "ablation_models"


def _safe_slug(value):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", value)


def run_ablation_tests(selected_features, baseline_correlation, ablation_epochs=5):
    results = []
    ABLATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for feature_name in selected_features:
        ablated_features = [feature for feature in selected_features if feature != feature_name]
        if not ablated_features:
            continue

        print(f"Running ablation without: {feature_name}")
        set_active_features(ablated_features, persist=False)
        model_path = ABLATION_MODEL_DIR / f"alpha_without_{_safe_slug(feature_name)}.pt"

        train_alpha_model(model_path=model_path, epochs=ablation_epochs)
        metrics = evaluate(model_path=model_path, return_metrics=True)
        performance_drop = baseline_correlation - metrics["correlation"]

        results.append(
            {
                "feature_name": feature_name,
                "performance_drop": performance_drop,
                "ablation_correlation": metrics["correlation"],
                "ablation_directional_accuracy": metrics["directional_accuracy"],
            }
        )

    set_active_features(selected_features, persist=True)

    ablation_df = pd.DataFrame(results)
    if not ablation_df.empty:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        ablation_df.to_csv(ABLATION_RESULTS_PATH, index=False)
    return ablation_df


def run_alpha_research(top_k=10, run_ablation=True, ablation_epochs=5):
    print("Starting alpha research pipeline...")
    reset_active_features(persist=False)

    train_df, _ = build_modeling_dataframes()
    if train_df.empty:
        raise RuntimeError("Training dataframe is empty; cannot run alpha research")

    feature_stats = evaluate_feature_importance(train_df)
    filtered_stats, dropped_stats = filter_features(feature_stats, train_df)
    selection_source = filtered_stats if not filtered_stats.empty else feature_stats
    selected_features = select_top_features(selection_source, top_k=top_k, persist=True)

    print("Selected features:", ", ".join(selected_features))
    top_ic_features = selection_source.head(min(10, len(selection_source)))["feature_name"].tolist()
    print("Top IC features:", ", ".join(top_ic_features))
    if dropped_stats.empty:
        print("Dropped features: none")
    else:
        dropped_summary = ", ".join(
            f"{row.feature_name} ({row.drop_reason})"
            for row in dropped_stats.itertuples(index=False)
        )
        print("Dropped features:", dropped_summary)

    train_metrics = train_alpha_model(return_metrics=True)
    eval_metrics = evaluate(return_metrics=True)

    research_rows = [
        {
            "selected_features": "|".join(selected_features),
            "feature_count": len(selected_features),
            "final_loss": train_metrics["final_loss"],
            "test_correlation": eval_metrics["correlation"],
            "test_directional_accuracy": eval_metrics["directional_accuracy"],
        }
    ]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(research_rows).to_csv(RESEARCH_RESULTS_PATH, index=False)

    ablation_df = pd.DataFrame()
    if run_ablation:
        ablation_df = run_ablation_tests(selected_features, eval_metrics["correlation"], ablation_epochs=ablation_epochs)

    return {
        "selected_features": selected_features,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "feature_stats": feature_stats,
        "dropped_features": dropped_stats,
        "ablation_results": ablation_df,
    }


if __name__ == "__main__":
    run_alpha_research()
