from pathlib import Path
import contextlib
import gc
import io
import random
import sys

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from torch.optim import Adam
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import (
    FEATURE_COLUMNS,
    GLOBAL_CONTEXT_FEATURES,
    REGIME_TO_ID,
    SEQUENCE_LENGTH,
    build_dataset_from_dataframe,
    get_active_features,
    load_training_dataframe,
    sanitize_training_dataframe,
    set_active_features,
    _normalize_per_timestamp,
    _recompute_targets_within_split,
)
from agents.alpha_agent.loss_functions import masked_mse_loss
from agents.alpha_agent.neuroquant_model import NeuroQuantAlphaModel
from agents.alpha_agent.run_structured_alpha_upgrade import (
    _engineer_research_features,
    _normalize_features,
    _select_top_features,
)
from agents.portfolio_agent.env_portfolio import PortfolioEnv


PROJECT_LOGS = PROJECT_ROOT / "logs"
RESULTS_PATH = PROJECT_LOGS / "walk_forward_results.csv"
TRAIN_FRACTION = 0.60
TEST_FRACTION = 0.20
STEP_FRACTION = 0.20
ALPHA_EPOCHS = 20
ALPHA_BATCH_SIZE = 1
ALPHA_LEARNING_RATE = 5e-4
PORTFOLIO_TIMESTEPS = 20000
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WalkForwardPortfolioEnv(PortfolioEnv):
    def __init__(self, split, train_df, test_df, active_features, alpha_model, reward_frame, **kwargs):
        self._wf_train_df = train_df.copy()
        self._wf_test_df = test_df.copy()
        self._wf_active_features = list(active_features)
        self._wf_alpha_model = alpha_model.to(torch.device("cpu"))
        self._wf_reward_frame = reward_frame.copy()
        super().__init__(split=split, **kwargs)

    def _resolve_active_features(self):
        self.research_train_df = self._wf_train_df.copy()
        self.research_test_df = self._wf_test_df.copy()
        return list(self._wf_active_features)

    def _load_alpha_model(self):
        self.alpha_device = torch.device("cpu")
        model = self._wf_alpha_model.to(self.alpha_device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _prepare_steps(self):
        split_df = self.research_train_df.copy() if self.split == "train" else self.research_test_df.copy()
        reward_frame = self._wf_reward_frame.copy()
        reward_frame["timestamp"] = pd.to_datetime(reward_frame["timestamp"])
        split_df = self._attach_next_returns(split_df, reward_frame)

        missing_active = [feature for feature in self.active_features if feature not in split_df.columns]
        if missing_active:
            raise RuntimeError(f"Missing portfolio observation features: {missing_active}")

        missing_alpha = [feature for feature in self.alpha_features if feature not in split_df.columns]
        if missing_alpha:
            raise RuntimeError(f"Missing alpha inference features: {missing_alpha}")

        symbol_windows = {}
        for symbol, group in split_df.groupby("symbol", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)
            if len(group) < SEQUENCE_LENGTH + 1:
                continue

            active_feature_matrix = group[self.active_features].to_numpy(dtype=np.float32, copy=False)
            alpha_feature_matrix = group[self.alpha_features].to_numpy(dtype=np.float32, copy=False)
            regime_series = group["regime_label"].map(REGIME_TO_ID).to_numpy(dtype=np.int64, copy=False)
            next_return_series = group["next_return"].to_numpy(dtype=np.float32, copy=False)
            timestamp_series = group["timestamp"].tolist()

            windows = {}
            for end_idx in range(SEQUENCE_LENGTH - 1, len(group) - 1):
                next_return = next_return_series[end_idx]
                if not np.isfinite(next_return):
                    continue

                start_idx = end_idx - SEQUENCE_LENGTH + 1
                timestamp = timestamp_series[end_idx]
                windows[timestamp] = {
                    "latest_features": active_feature_matrix[end_idx],
                    "alpha_sequence": alpha_feature_matrix[start_idx : end_idx + 1],
                    "regime_id": regime_series[end_idx],
                    "next_return": float(next_return),
                }

            if windows:
                symbol_windows[symbol] = windows

        if not symbol_windows:
            raise RuntimeError(f"No rolling windows available for split '{self.split}'")

        sorted_symbols = sorted(symbol_windows, key=lambda sym: len(symbol_windows[sym]), reverse=True)
        selected_symbols = []
        common_timestamps = None

        for symbol in sorted_symbols:
            symbol_timestamps = set(symbol_windows[symbol].keys())
            proposed_timestamps = symbol_timestamps if common_timestamps is None else common_timestamps & symbol_timestamps

            if len(proposed_timestamps) < self.min_timestamps:
                if not selected_symbols and len(symbol_timestamps) >= self.min_timestamps:
                    selected_symbols.append(symbol)
                    common_timestamps = symbol_timestamps
                continue

            selected_symbols.append(symbol)
            common_timestamps = proposed_timestamps
            if len(selected_symbols) >= self.max_assets:
                break

        if not selected_symbols:
            raise RuntimeError("Could not select a stable asset universe for the portfolio environment")

        if common_timestamps is None:
            common_timestamps = set(symbol_windows[selected_symbols[0]].keys())

        timeline = sorted(common_timestamps)
        if len(timeline) < 2:
            raise RuntimeError("Not enough aligned timestamps for portfolio training")

        steps = []
        for timestamp in timeline:
            latest_features = np.stack(
                [symbol_windows[symbol][timestamp]["latest_features"] for symbol in selected_symbols],
                axis=0,
            ).astype(np.float32)
            alpha_sequences = np.stack(
                [symbol_windows[symbol][timestamp]["alpha_sequence"] for symbol in selected_symbols],
                axis=0,
            ).astype(np.float32)
            regime_ids = np.asarray(
                [symbol_windows[symbol][timestamp]["regime_id"] for symbol in selected_symbols],
                dtype=np.int64,
            )
            next_returns = np.asarray(
                [symbol_windows[symbol][timestamp]["next_return"] for symbol in selected_symbols],
                dtype=np.float32,
            )
            alpha = self._predict_alpha(alpha_sequences, regime_ids)

            steps.append(
                {
                    "timestamp": timestamp,
                    "latest_features": latest_features,
                    "alpha": alpha.astype(np.float32),
                    "regime_ids": regime_ids,
                    "next_returns": next_returns,
                }
            )

        return selected_symbols, steps


def generate_walk_forward_windows(timestamps):
    total = len(timestamps)
    train_size = int(total * TRAIN_FRACTION)
    test_size = int(total * TEST_FRACTION)
    step_size = int(total * STEP_FRACTION)
    windows = []
    split_id = 1

    for start_idx in range(0, total - train_size - test_size + 1, step_size):
        train_start = timestamps[start_idx]
        train_end = timestamps[start_idx + train_size - 1]
        test_start = timestamps[start_idx + train_size]
        test_end = timestamps[start_idx + train_size + test_size - 1]
        windows.append(
            {
                "split_id": split_id,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        split_id += 1

    return windows


def prepare_alpha_frames(raw_df, train_start, train_end, test_start, test_end):
    train_df = raw_df[(raw_df["timestamp"] >= train_start) & (raw_df["timestamp"] <= train_end)].copy()
    test_df = raw_df[(raw_df["timestamp"] >= test_start) & (raw_df["timestamp"] <= test_end)].copy()

    train_df = _recompute_targets_within_split(train_df)
    test_df = _recompute_targets_within_split(test_df)

    if train_df.empty or test_df.empty:
        raise RuntimeError("Alpha walk-forward split produced an empty train/test frame")

    train_features = train_df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    train_fill_values = train_features.median(numeric_only=True).fillna(0.0)
    train_feature_frame = train_features.fillna(train_fill_values)
    global_means = train_feature_frame[GLOBAL_CONTEXT_FEATURES].mean().fillna(0.0)
    global_stds = train_feature_frame[GLOBAL_CONTEXT_FEATURES].std().replace(0, np.nan).fillna(1.0)

    train_df = _normalize_per_timestamp(train_df, train_fill_values, global_means, global_stds)
    test_df = _normalize_per_timestamp(test_df, train_fill_values, global_means, global_stds)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_portfolio_frames(raw_df, alpha_features, train_start, train_end, test_start, test_end):
    subset = raw_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["future_return_10", "regime_label"]).copy()
    train_df = subset[(subset["timestamp"] >= train_start) & (subset["timestamp"] <= train_end)].copy()
    test_df = subset[(subset["timestamp"] >= test_start) & (subset["timestamp"] <= test_end)].copy()

    if train_df.empty or test_df.empty:
        raise RuntimeError("Portfolio walk-forward split produced an empty train/test frame")

    selected_features = _select_top_features(train_df)
    observation_features = list(dict.fromkeys(selected_features + list(alpha_features)))
    train_df, test_df = _normalize_features(train_df, test_df, observation_features)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), observation_features


def train_alpha_model_for_split(train_df, alpha_features):
    original_features = get_active_features()
    set_active_features(alpha_features, persist=False)
    try:
        train_dataset = build_dataset_from_dataframe(train_df)
        if len(train_dataset) == 0:
            raise RuntimeError("Alpha training dataset is empty for this split")

        dataloader = DataLoader(
            train_dataset,
            batch_size=ALPHA_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = NeuroQuantAlphaModel(feature_dim=len(alpha_features)).to(device)
        optimizer = Adam(model.parameters(), lr=ALPHA_LEARNING_RATE)

        for _epoch in range(ALPHA_EPOCHS):
            model.train()
            for features, regime_ids, targets, masks in dataloader:
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
                regime_ids = regime_ids.to(device, non_blocking=True)
                targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0).to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                predicted_scores = model(features, regime_ids)
                loss = masked_mse_loss(predicted_scores, targets, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        return model.to(torch.device("cpu"))
    finally:
        set_active_features(original_features, persist=False)


def train_and_evaluate_portfolio(train_df, test_df, observation_features, alpha_model, reward_frame):
    train_env = WalkForwardPortfolioEnv(
        split="train",
        train_df=train_df,
        test_df=test_df,
        active_features=observation_features,
        alpha_model=alpha_model,
        reward_frame=reward_frame,
    )
    model = PPO("MlpPolicy", train_env, verbose=0, seed=SEED)
    model.learn(total_timesteps=PORTFOLIO_TIMESTEPS)

    test_env = WalkForwardPortfolioEnv(
        split="test",
        train_df=train_df,
        test_df=test_df,
        active_features=observation_features,
        alpha_model=alpha_model,
        reward_frame=reward_frame,
    )

    observation, _info = test_env.reset()
    done = False
    final_info = {}
    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        final_info = info

    return {
        "final_return": float(final_info.get("final_return", 0.0)),
        "sharpe": float(final_info.get("final_sharpe", 0.0)),
        "drawdown": float(final_info.get("max_drawdown", 0.0)),
        "turnover": float(final_info.get("final_turnover", 0.0)),
    }


def summarize_metrics(results_df):
    summary = {}
    for metric_name in ["final_return", "sharpe", "drawdown", "turnover"]:
        summary[f"avg_{metric_name}"] = float(results_df[metric_name].mean())
        summary[f"std_{metric_name}"] = float(results_df[metric_name].std(ddof=0)) if len(results_df) > 1 else 0.0
    return summary


def main():
    set_seed(SEED)
    PROJECT_LOGS.mkdir(parents=True, exist_ok=True)

    alpha_features = get_active_features()

    with contextlib.redirect_stdout(io.StringIO()):
        raw_alpha_df = sanitize_training_dataframe(load_training_dataframe())
        raw_portfolio_df = _engineer_research_features(load_training_dataframe())

    raw_alpha_df["timestamp"] = pd.to_datetime(raw_alpha_df["timestamp"])
    raw_portfolio_df["timestamp"] = pd.to_datetime(raw_portfolio_df["timestamp"])

    reward_frame = raw_alpha_df[["symbol", "timestamp", "log_return_1"]].copy()
    reward_frame = reward_frame.rename(columns={"log_return_1": "realized_return_1"})

    timestamps = sorted(raw_portfolio_df["timestamp"].dropna().unique())
    windows = generate_walk_forward_windows(timestamps)
    if not windows:
        raise RuntimeError("No walk-forward windows could be generated from the available timestamps")

    rows = []
    for window in windows:
        print(
            f"Running split {window['split_id']}: "
            f"train={window['train_start'].date()}->{window['train_end'].date()}, "
            f"test={window['test_start'].date()}->{window['test_end'].date()}"
        )

        train_alpha_df, test_alpha_df = prepare_alpha_frames(
            raw_alpha_df,
            window["train_start"],
            window["train_end"],
            window["test_start"],
            window["test_end"],
        )
        alpha_model = train_alpha_model_for_split(train_alpha_df, alpha_features)

        train_portfolio_df, test_portfolio_df, observation_features = prepare_portfolio_frames(
            raw_portfolio_df,
            alpha_features,
            window["train_start"],
            window["train_end"],
            window["test_start"],
            window["test_end"],
        )
        metrics = train_and_evaluate_portfolio(
            train_portfolio_df,
            test_portfolio_df,
            observation_features,
            alpha_model,
            reward_frame,
        )

        row = {
            "split_id": window["split_id"],
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            **metrics,
        }
        rows.append(row)
        print(
            f"SPLIT_{window['split_id']}_RETURN={metrics['final_return']:.6f}\n"
            f"SPLIT_{window['split_id']}_SHARPE={metrics['sharpe']:.6f}\n"
            f"SPLIT_{window['split_id']}_DRAWDOWN={metrics['drawdown']:.6f}\n"
            f"SPLIT_{window['split_id']}_TURNOVER={metrics['turnover']:.6f}"
        )

        del alpha_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_PATH, index=False)

    summary = summarize_metrics(results_df)
    print(f"AVG_FINAL_RETURN={summary['avg_final_return']:.6f}")
    print(f"STD_FINAL_RETURN={summary['std_final_return']:.6f}")
    print(f"AVG_SHARPE={summary['avg_sharpe']:.6f}")
    print(f"STD_SHARPE={summary['std_sharpe']:.6f}")
    print(f"AVG_DRAWDOWN={summary['avg_drawdown']:.6f}")
    print(f"STD_DRAWDOWN={summary['std_drawdown']:.6f}")
    print(f"AVG_TURNOVER={summary['avg_turnover']:.6f}")
    print(f"STD_TURNOVER={summary['std_turnover']:.6f}")


if __name__ == "__main__":
    main()
