from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.portfolio_agent.env_portfolio import PortfolioEnv


MODEL_PATH = PROJECT_ROOT / "models" / "portfolio_models" / "ppo_model.zip"
PLOT_PATH = PROJECT_ROOT / "logs" / "portfolio_analysis.png"
ROLLING_WINDOW = 50


def run_episode(model_path=MODEL_PATH):
    env = PortfolioEnv(split="test")
    model = PPO.load(str(model_path), env=env)

    observation, _info = env.reset()
    done = False
    records = []
    final_info = {}

    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(action)
        weights = np.asarray(info.get("weights", []), dtype=np.float32)

        records.append(
            {
                "timestamp": pd.to_datetime(info.get("timestamp")),
                "portfolio_return": float(info.get("portfolio_return", 0.0)),
                "turnover": float(info.get("turnover", 0.0)),
                "long_exposure": float(weights[weights > 0].sum()) if weights.size else 0.0,
                "short_exposure": float(weights[weights < 0].sum()) if weights.size else 0.0,
                "weights": weights.copy(),
            }
        )

        done = terminated or truncated
        final_info = info

    return env, pd.DataFrame(records), final_info


def build_metrics(frame):
    frame = frame.copy()
    returns = frame["portfolio_return"].astype(float)

    frame["wealth"] = (1.0 + returns).cumprod()
    frame["peak"] = frame["wealth"].cummax()
    frame["drawdown"] = 1.0 - frame["wealth"] / (frame["peak"] + 1e-8)

    rolling_mean = returns.rolling(ROLLING_WINDOW, min_periods=2).mean()
    rolling_std = returns.rolling(ROLLING_WINDOW, min_periods=2).std()
    frame["rolling_sharpe"] = rolling_mean / (rolling_std + 1e-6)

    final_return = float(frame["wealth"].iloc[-1] - 1.0) if not frame.empty else 0.0
    final_sharpe = float(returns.mean() / (returns.std() + 1e-6)) if len(frame) > 1 else 0.0
    max_drawdown = float(frame["drawdown"].max()) if not frame.empty else 0.0

    return frame, {
        "final_return": final_return,
        "final_sharpe": final_sharpe,
        "max_drawdown": max_drawdown,
    }


def plot_metrics(frame, plot_path=PLOT_PATH):
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)

    axes[0].plot(frame["timestamp"], frame["wealth"], color="#0f766e", linewidth=1.8)
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Wealth")
    axes[0].grid(alpha=0.3)

    axes[1].plot(frame["timestamp"], frame["drawdown"], color="#b91c1c", linewidth=1.5)
    axes[1].fill_between(frame["timestamp"], 0.0, frame["drawdown"], color="#fecaca", alpha=0.7)
    axes[1].set_title("Drawdown Curve")
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(alpha=0.3)

    axes[2].plot(frame["timestamp"], frame["rolling_sharpe"], color="#1d4ed8", linewidth=1.5)
    axes[2].axhline(0.0, color="#64748b", linewidth=1.0, linestyle="--")
    axes[2].set_title(f"Rolling Sharpe ({ROLLING_WINDOW})")
    axes[2].set_ylabel("Sharpe")
    axes[2].grid(alpha=0.3)

    axes[3].plot(frame["timestamp"], frame["long_exposure"], color="#15803d", linewidth=1.5, label="Long Exposure")
    axes[3].plot(frame["timestamp"], frame["short_exposure"], color="#7c3aed", linewidth=1.5, label="Short Exposure")
    axes[3].axhline(0.0, color="#64748b", linewidth=1.0, linestyle="--")
    axes[3].set_title("Long vs Short Exposure")
    axes[3].set_ylabel("Exposure")
    axes[3].legend(loc="upper right")
    axes[3].grid(alpha=0.3)

    axes[4].plot(frame["timestamp"], frame["turnover"], color="#ea580c", linewidth=1.5)
    axes[4].set_title("Turnover Over Time")
    axes[4].set_ylabel("Turnover")
    axes[4].set_xlabel("Timestamp")
    axes[4].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main():
    _env, frame, _final_info = run_episode()
    analytics, metrics = build_metrics(frame)
    plot_metrics(analytics)

    print(f"FINAL_RETURN={metrics['final_return']:.6f}")
    print(f"FINAL_SHARPE={metrics['final_sharpe']:.6f}")
    print(f"MAX_DRAWDOWN={metrics['max_drawdown']:.6f}")


if __name__ == "__main__":
    main()
