"""Train the from-scratch DQN agent and write episode metrics to CSV."""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from .config import DQNConfig, EnvConfig, ScenarioSpec, resolve_scenario
from .dqn import DQNAgent
from .environment import DrivingEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", choices=("lane", "traffic"), default="traffic")
    parser.add_argument(
        "--scenario-preset",
        choices=("sparse", "normal", "dense", "random"),
        default="random",
        help="world conditions for full-traffic episodes (default: random)",
    )
    parser.add_argument("--traffic", type=int, default=None, help="override car count")
    parser.add_argument("--obstacles", type=int, default=None, help="override obstacle count")
    parser.add_argument("--reactive", type=float, default=None, help="override reactive fraction 0..1")
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="begin with lane keeping before introducing traffic (default: on)",
    )
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autodrive_dqn.npz"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("runs/training_metrics.csv"),
    )
    return parser


def evaluate(
    agent: DQNAgent,
    env_config: EnvConfig,
    *,
    episodes: int,
    seed: int,
    scenario: str = "traffic",
) -> dict[str, float]:
    returns: list[float] = []
    distances: list[float] = []
    safe_finishes = 0
    for episode_index in range(episodes):
        env = DrivingEnv(env_config, scenario=scenario, seed=seed + episode_index)
        observation, _ = env.reset(seed=seed + episode_index)
        episode_return = 0.0
        while True:
            action = agent.act(observation, explore=False)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            if terminated or truncated:
                if not info["collision"] and not info["off_road"]:
                    safe_finishes += 1
                returns.append(episode_return)
                distances.append(float(info["distance_m"]))
                break
    return {
        "return": fmean(returns),
        "distance_m": fmean(distances),
        "safe_rate": safe_finishes / episodes,
    }


def train(
    *,
    episodes: int,
    env_config: EnvConfig,
    dqn_config: DQNConfig | None = None,
    seed: int = 7,
    scenario: str = "traffic",
    curriculum: bool = True,
    scenario_preset: str = "random",
    traffic: int | None = None,
    obstacles: int | None = None,
    reactive: float | None = None,
    eval_every: int = 25,
    eval_episodes: int = 3,
    log_every: int = 5,
    output_path: Path = Path("models/autodrive_dqn.npz"),
    metrics_path: Path = Path("runs/training_metrics.csv"),
) -> tuple[DQNAgent, list[dict[str, Any]]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")

    agent = DQNAgent(
        DrivingEnv.observation_size,
        DrivingEnv.action_size,
        dqn_config,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_path = output_path.with_name(f"{output_path.stem}_best{output_path.suffix}")
    best_eval_return = -np.inf
    recent_returns: deque[float] = deque(maxlen=20)
    records: list[dict[str, Any]] = []

    lane_curriculum_end = (
        max(5, int(episodes * 0.15)) if curriculum and scenario == "traffic" else 0
    )
    light_traffic_end = (
        max(lane_curriculum_end + 5, int(episodes * 0.35))
        if curriculum and scenario == "traffic"
        else 0
    )
    scenario_rng = np.random.default_rng(seed + 777_777)

    for episode in range(1, episodes + 1):
        episode_spec: ScenarioSpec | None = None
        if episode <= lane_curriculum_end:
            env_scenario = "lane"
            episode_scenario = "lane"
            episode_config = env_config
        elif episode <= light_traffic_end:
            env_scenario = "traffic"
            episode_scenario = "light_traffic"
            episode_config = replace(
                env_config,
                traffic_count=max(3, env_config.traffic_count // 2),
            )
        else:
            env_scenario = scenario
            episode_scenario = scenario
            episode_config = env_config
            if scenario == "traffic":
                episode_spec = resolve_scenario(
                    scenario_preset,
                    traffic=traffic,
                    obstacles=obstacles,
                    reactive=reactive,
                    rng=scenario_rng,
                )
        env = DrivingEnv(
            episode_config, scenario=env_scenario, seed=seed + episode, scenario_spec=episode_spec
        )
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_losses: list[float] = []
        speed_sum = 0.0
        final_info: dict[str, Any] = {}

        while True:
            action = agent.act(observation, explore=True)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.observe(observation, action, reward, next_observation, done)
            learn_result = agent.learn()
            if learn_result is not None:
                episode_losses.append(learn_result.loss)
            observation = next_observation
            episode_return += reward
            speed_sum += float(info["speed_mps"])
            final_info = info
            if done:
                break

        recent_returns.append(episode_return)
        if final_info["collision"]:
            outcome = "collision"
        elif final_info["off_road"]:
            outcome = "off_road"
        else:
            outcome = "complete"

        record: dict[str, Any] = {
            "episode": episode,
            "scenario": episode_scenario,
            "traffic_count": sum(1 for car in env.traffic if car.behavior != "obstacle"),
            "obstacle_count": sum(1 for car in env.traffic if car.behavior == "obstacle"),
            "reactive_fraction": round(episode_spec.reactive_fraction, 4)
            if episode_spec is not None
            else 0.0,
            "steps": final_info["steps"],
            "return": round(episode_return, 5),
            "rolling_return_20": round(fmean(recent_returns), 5),
            "distance_m": round(float(final_info["distance_m"]), 3),
            "mean_speed_mps": round(speed_sum / int(final_info["steps"]), 4),
            "outcome": outcome,
            "epsilon": round(agent.epsilon, 6),
            "mean_loss": round(fmean(episode_losses), 6) if episode_losses else "",
            "eval_return": "",
            "eval_distance_m": "",
            "eval_safe_rate": "",
        }

        should_evaluate = eval_every > 0 and (episode % eval_every == 0 or episode == episodes)
        if should_evaluate:
            evaluation = evaluate(
                agent,
                env_config,
                episodes=eval_episodes,
                seed=seed + 100_000 + episode * eval_episodes,
                scenario=scenario,
            )
            record["eval_return"] = round(evaluation["return"], 5)
            record["eval_distance_m"] = round(evaluation["distance_m"], 3)
            record["eval_safe_rate"] = round(evaluation["safe_rate"], 4)
            if evaluation["return"] > best_eval_return:
                best_eval_return = evaluation["return"]
                agent.save(best_path)

        records.append(record)
        if episode % max(1, log_every) == 0 or episode == 1 or episode == episodes:
            eval_text = (
                f" eval={float(record['eval_return']):7.1f}"
                if record["eval_return"] != ""
                else ""
            )
            print(
                f"episode {episode:4d}/{episodes}  {episode_scenario:7s}  "
                f"return={episode_return:8.1f}  avg20={fmean(recent_returns):8.1f}  "
                f"steps={int(final_info['steps']):4d}  eps={agent.epsilon:.3f}  "
                f"outcome={outcome}{eval_text}"
            )

    agent.save(output_path)
    _write_metrics(metrics_path, records)
    print(f"\nSaved final model: {output_path}")
    print(f"Saved best model:  {best_path}")
    print(f"Saved metrics:     {metrics_path}")
    return agent, records


def _write_metrics(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    env_config = replace(EnvConfig(), max_steps=args.max_steps)
    train(
        episodes=args.episodes,
        env_config=env_config,
        seed=args.seed,
        scenario=args.scenario,
        curriculum=args.curriculum,
        scenario_preset=args.scenario_preset,
        traffic=args.traffic,
        obstacles=args.obstacles,
        reactive=args.reactive,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        log_every=args.log_every,
        output_path=args.output,
        metrics_path=args.metrics,
    )


if __name__ == "__main__":
    main()
