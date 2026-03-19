from pathlib import Path
import contextlib
import io
import sys

import numpy as np
import pandas as pd
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.alpha_agent.dataset_builder import (
    REGIME_TO_ID,
    SEQUENCE_LENGTH,
    get_active_features,
    load_training_dataframe,
    sanitize_training_dataframe,
    split_and_normalize_dataframe,
)
from agents.alpha_agent.neuroquant_model import NeuroQuantAlphaModel


DEFAULT_MAX_ASSETS = 16
DEFAULT_MAX_WEIGHT = 0.10
DEFAULT_TURNOVER_PENALTY = 0.05
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_ROLLING_VOL_WINDOW = 20
DEFAULT_MIN_TIMESTEPS = 64
ALPHA_MODEL_PATH = PROJECT_ROOT / "models" / "neuroquant_alpha_model.pt"


def equal_weight_portfolio(num_assets):
    if num_assets <= 0:
        return np.asarray([], dtype=np.float32)
    return np.full(num_assets, 1.0 / num_assets, dtype=np.float32)


def flat_portfolio(num_assets):
    if num_assets <= 0:
        return np.asarray([], dtype=np.float32)
    return np.zeros(num_assets, dtype=np.float32)


def _portfolio_path_stats(returns):
    if not returns:
        return 0.0, 0.0

    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=np.float64))
    peaks = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - (wealth / np.maximum(peaks, 1e-8))
    cumulative_return = float(wealth[-1] - 1.0)
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0
    return cumulative_return, max_drawdown


def _squash_alpha_signal(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.where(np.isfinite(values), values, 0.0)
    return np.tanh(values).astype(np.float32)


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        split="train",
        max_assets=DEFAULT_MAX_ASSETS,
        max_weight=DEFAULT_MAX_WEIGHT,
        turnover_penalty=DEFAULT_TURNOVER_PENALTY,
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
        rolling_vol_window=DEFAULT_ROLLING_VOL_WINDOW,
        min_timestamps=DEFAULT_MIN_TIMESTEPS,
    ):
        super().__init__()
        self.split = split
        self.max_assets = max_assets
        self.max_weight = max_weight
        self.turnover_penalty = turnover_penalty
        self.risk_free_rate = risk_free_rate
        self.rolling_vol_window = rolling_vol_window
        self.min_timestamps = min_timestamps
        self.research_train_df = None
        self.research_test_df = None
        self.active_features = self._resolve_active_features()
        self.alpha_features = get_active_features()
        self.alpha_device = torch.device("cpu")
        self.alpha_model = self._load_alpha_model()

        self.asset_symbols, self.steps = self._prepare_steps()
        self.num_assets = len(self.asset_symbols)

        if self.num_assets == 0 or not self.steps:
            raise RuntimeError("Portfolio environment could not be initialized with valid assets/timesteps")

        self.observation_dim = (self.num_assets * len(self.active_features)) + (2 * self.num_assets)
        self.action_dim = self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.current_step = 0
        self.previous_weights = flat_portfolio(self.num_assets)
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_turnovers = []

    def _resolve_active_features(self):
        try:
            from agents.alpha_agent.run_structured_alpha_upgrade import build_research_frames

            with contextlib.redirect_stdout(io.StringIO()):
                train_df, test_df, selected_features = build_research_frames()
            self.research_train_df = train_df.copy()
            self.research_test_df = test_df.copy()
            return selected_features
        except Exception:
            return get_active_features()

    def _load_alpha_model(self):
        if not ALPHA_MODEL_PATH.exists():
            raise RuntimeError(f"Alpha model not found at {ALPHA_MODEL_PATH}")

        model = NeuroQuantAlphaModel(feature_dim=len(self.alpha_features)).to(self.alpha_device)
        state_dict = torch.load(ALPHA_MODEL_PATH, map_location=self.alpha_device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    @staticmethod
    def _attach_next_returns(split_df, reward_frame):
        split_df = split_df.merge(reward_frame, on=["symbol", "timestamp"], how="left")
        split_df["next_return"] = split_df.groupby("symbol", sort=False)["realized_return_1"].shift(-1)
        split_df = split_df.drop(columns=["realized_return_1"])
        return split_df

    def _predict_alpha(self, alpha_sequences, regime_ids):
        with torch.no_grad():
            feature_tensor = torch.from_numpy(alpha_sequences).unsqueeze(0).to(self.alpha_device)
            regime_tensor = torch.from_numpy(regime_ids).unsqueeze(0).to(self.alpha_device)
            alpha_scores = self.alpha_model(feature_tensor, regime_tensor).squeeze(0).cpu().numpy()
        return _squash_alpha_signal(alpha_scores)

    def _prepare_steps(self):
        raw_df = sanitize_training_dataframe(load_training_dataframe())
        reward_frame = raw_df[["symbol", "timestamp", "log_return_1"]].copy()
        reward_frame = reward_frame.rename(columns={"log_return_1": "realized_return_1"})
        reward_frame["timestamp"] = pd.to_datetime(reward_frame["timestamp"])

        if self.research_train_df is not None and self.research_test_df is not None:
            train_df = self.research_train_df.copy()
            test_df = self.research_test_df.copy()
        else:
            train_df, test_df = split_and_normalize_dataframe(raw_df)

        train_df = self._attach_next_returns(train_df, reward_frame)
        test_df = self._attach_next_returns(test_df, reward_frame)
        split_df = train_df if self.split == "train" else test_df

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

    def _normalize_side(self, signal, target_total=0.5):
        signal = np.asarray(signal, dtype=np.float64)
        signal = np.where(np.isfinite(signal), signal, 0.0)
        signal = np.clip(signal, 0.0, None)

        active = signal > 0.0
        if not active.any() or target_total <= 0.0:
            return np.zeros_like(signal, dtype=np.float32)

        side = np.zeros_like(signal, dtype=np.float64)
        active_signal = signal[active].copy()
        active_weights = active_signal / active_signal.sum() * target_total

        for _ in range(active_weights.size * 4):
            over = active_weights > self.max_weight + 1e-10
            if not over.any():
                break

            excess = float((active_weights[over] - self.max_weight).sum())
            active_weights[over] = self.max_weight
            under = ~over
            if excess <= 0.0 or not under.any():
                break

            active_under_signal = active_signal[under]
            signal_sum = float(active_under_signal.sum())
            if signal_sum <= 0.0:
                break
            active_weights[under] += excess * (active_under_signal / signal_sum)

        active_weights = np.clip(active_weights, 0.0, self.max_weight)
        side[active] = active_weights
        return side.astype(np.float32)

    def _normalize_signed_weights(self, weights, center=False):
        weights = np.asarray(weights, dtype=np.float64)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if center:
            weights = weights - float(weights.mean())

        positive_side = self._normalize_side(np.clip(weights, 0.0, None), target_total=0.5)
        negative_side = self._normalize_side(np.clip(-weights, 0.0, None), target_total=0.5)
        combined = positive_side - negative_side
        combined = np.clip(combined, -self.max_weight, self.max_weight)
        return combined.astype(np.float32)

    def _get_observation(self):
        step = self.steps[self.current_step]
        return np.concatenate(
            [
                step["latest_features"].reshape(-1),
                step["alpha"],
                step["regime_ids"].astype(np.float32),
            ]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_weights = flat_portfolio(self.num_assets)
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_turnovers = []

        observation = self._get_observation()
        info = {
            "timestamp": str(self.steps[self.current_step]["timestamp"]),
            "symbols": self.asset_symbols,
            "features": self.steps[self.current_step]["latest_features"].copy(),
            "alpha": self.steps[self.current_step]["alpha"].copy(),
            "regime_state": self.steps[self.current_step]["regime_ids"].copy(),
        }
        return observation, info

    def step(self, action):
        action_weights = self._normalize_signed_weights(action, center=True)
        step = self.steps[self.current_step]

        alpha = np.asarray(step["alpha"], dtype=np.float32)
        alpha = np.where(np.isfinite(alpha), alpha, 0.0)

        top_k = max(2, int(0.2 * self.num_assets))
        sorted_idx = np.argsort(alpha)
        short_idx = sorted_idx[:top_k]
        long_idx = sorted_idx[-top_k:]

        alpha_signal = np.zeros(self.num_assets, dtype=np.float32)
        alpha_signal[long_idx] = np.clip(alpha[long_idx], 0.0, None)
        alpha_signal[short_idx] = -np.clip(-alpha[short_idx], 0.0, None)
        alpha_weights = self._normalize_signed_weights(alpha_signal)

        blended_weights = 0.8 * alpha_weights + 0.2 * action_weights
        weights = 0.8 * self.previous_weights + 0.2 * blended_weights
        weights = self._normalize_signed_weights(weights)

        asset_returns = step["next_returns"]
        portfolio_return = float(np.dot(weights, asset_returns))
        turnover = 0.5 * float(np.sum(np.abs(weights - self.previous_weights)))

        self.episode_returns.append(portfolio_return)
        window = min(len(self.episode_returns), 50)
        recent_returns = np.asarray(self.episode_returns[-window:], dtype=np.float32)

        mean_return = float(recent_returns.mean()) if recent_returns.size else 0.0
        volatility = float(recent_returns.std()) + 1e-6
        sharpe = mean_return / volatility

        cumulative = np.cumprod(1.0 + recent_returns.astype(np.float64)) if recent_returns.size else np.asarray([], dtype=np.float64)
        peak = np.maximum.accumulate(cumulative) if cumulative.size else np.asarray([], dtype=np.float64)
        drawdown = float(np.max(1.0 - cumulative / (peak + 1e-8))) if cumulative.size else 0.0

        reward = sharpe
        reward -= 0.5 * drawdown
        reward -= self.turnover_penalty * turnover

        self.episode_rewards.append(reward)
        self.episode_turnovers.append(turnover)
        self.previous_weights = weights

        self.current_step += 1
        terminated = self.current_step >= len(self.steps)
        truncated = False

        if terminated:
            next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            next_step_payload = None
        else:
            next_observation = self._get_observation()
            next_step_payload = self.steps[self.current_step]

        average_return = float(np.mean(self.episode_returns)) if self.episode_returns else 0.0
        episode_volatility = float(np.std(self.episode_returns)) if len(self.episode_returns) > 1 else 0.0
        sharpe_ratio = (average_return - self.risk_free_rate) / (episode_volatility + 1e-6) if self.episode_returns else 0.0
        cumulative_return, max_drawdown = _portfolio_path_stats(self.episode_returns)
        average_turnover = float(np.mean(self.episode_turnovers)) if self.episode_turnovers else 0.0

        info = {
            "portfolio_return": portfolio_return,
            "average_return": average_return,
            "volatility": episode_volatility,
            "sharpe_ratio": sharpe_ratio,
            "turnover": turnover,
            "max_drawdown": max_drawdown,
            "weights": weights,
            "timestamp": str(step["timestamp"]),
        }

        if next_step_payload is not None:
            info["features"] = next_step_payload["latest_features"].copy()
            info["alpha"] = next_step_payload["alpha"].copy()
            info["regime_state"] = next_step_payload["regime_ids"].copy()

        if terminated:
            info["episode_reward"] = float(np.sum(self.episode_rewards))
            info["final_return"] = cumulative_return
            info["final_volatility"] = episode_volatility
            info["final_sharpe"] = sharpe_ratio
            info["max_drawdown"] = max_drawdown
            info["final_turnover"] = average_turnover

        return next_observation, reward, terminated, truncated, info

    def render(self):
        return None
