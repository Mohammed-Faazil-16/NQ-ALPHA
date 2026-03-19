from pathlib import Path
import sys

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.portfolio_agent.env_portfolio import PortfolioEnv


TOTAL_TIMESTEPS = 20000
MODEL_STEM = PROJECT_ROOT / "models" / "portfolio_models" / "ppo_model"
MODEL_ZIP = MODEL_STEM.with_suffix(".zip")


def train_portfolio_model(model_stem=MODEL_STEM, total_timesteps=TOTAL_TIMESTEPS):
    env = PortfolioEnv(split="train")
    model = PPO("MlpPolicy", env, verbose=1, seed=42)
    model.learn(total_timesteps=total_timesteps)
    model_stem.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_stem))
    return model


def evaluate_portfolio(model=None, model_stem=MODEL_STEM):
    env = PortfolioEnv(split="test")
    if model is None:
        model = PPO.load(str(model_stem), env=env)

    observation, _info = env.reset()
    done = False
    final_info = {}

    while not done:
        action, _state = model.predict(observation, deterministic=True)
        observation, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        final_info = info

    return {
        "final_return": float(final_info.get("final_return", 0.0)),
        "final_sharpe": float(final_info.get("final_sharpe", 0.0)),
        "max_drawdown": float(final_info.get("max_drawdown", 0.0)),
        "turnover": float(final_info.get("final_turnover", 0.0)),
    }


def main():
    model = train_portfolio_model()
    metrics = evaluate_portfolio(model=model)

    print(f"FINAL_RETURN={metrics['final_return']:.6f}")
    print(f"FINAL_SHARPE={metrics['final_sharpe']:.6f}")
    print(f"MAX_DRAWDOWN={metrics['max_drawdown']:.6f}")
    print(f"TURNOVER={metrics['turnover']:.6f}")


if __name__ == "__main__":
    main()
