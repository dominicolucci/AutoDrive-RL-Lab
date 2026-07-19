"""Run the visual simulator with manual, heuristic, random, or DQN control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import resolve_scenario
from .dqn import DQNAgent
from .environment import Action, DrivingEnv
from .heuristic import HeuristicDriver
from .renderer import TopDownRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        choices=("heuristic", "manual", "random", "dqn"),
        default="heuristic",
        help="who controls the ego car (default: heuristic)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/autodrive_dqn.npz"),
        help="saved .npz model used by --policy dqn",
    )
    parser.add_argument("--scenario", choices=("lane", "traffic"), default="traffic")
    parser.add_argument(
        "--scenario-preset",
        choices=("sparse", "normal", "dense", "random"),
        default="normal",
        help="world conditions to drive in (default: normal, today's world)",
    )
    parser.add_argument("--traffic", type=int, default=None, help="override car count")
    parser.add_argument("--obstacles", type=int, default=None, help="override obstacle count")
    parser.add_argument("--reactive", type=float, default=None, help="override reactive fraction 0..1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fps", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scenario_spec = resolve_scenario(
        args.scenario_preset,
        traffic=args.traffic,
        obstacles=args.obstacles,
        reactive=args.reactive,
        rng=np.random.default_rng(args.seed),
    )
    env = DrivingEnv(scenario=args.scenario, seed=args.seed, scenario_spec=scenario_spec)
    renderer = TopDownRenderer(fps=args.fps)
    heuristic = HeuristicDriver()
    agent: DQNAgent | None = None
    if args.policy == "dqn":
        if not args.model.exists():
            raise SystemExit(
                f"Model not found: {args.model}. Train it first with "
                "`python -m autodrive_rl.train --episodes 300`."
            )
        agent = DQNAgent.load(args.model, seed=args.seed)
        if agent.observation_size != env.observation_size or agent.action_size != env.action_size:
            raise SystemExit("The saved model does not match this environment version.")

    observation, _ = env.reset(seed=args.seed)
    episode = 1
    episode_reward = 0.0
    action = int(Action.MAINTAIN)
    terminal_message: str | None = None
    terminal_frames = 0

    try:
        while not renderer.closed:
            renderer.process_events()
            if renderer.closed:
                break
            if renderer.reset_requested:
                renderer.reset_requested = False
                observation, _ = env.reset(seed=args.seed + episode)
                heuristic.reset()
                episode_reward = 0.0
                terminal_message = None
                terminal_frames = 0

            if terminal_frames > 0:
                terminal_frames -= 1
                if terminal_frames == 0:
                    episode += 1
                    observation, _ = env.reset(seed=args.seed + episode)
                    heuristic.reset()
                    episode_reward = 0.0
                    terminal_message = None
            elif not renderer.paused:
                if args.policy == "manual":
                    action = renderer.manual_action()
                elif args.policy == "heuristic":
                    action = heuristic.act(env)
                elif args.policy == "random":
                    action = int(env.rng.integers(0, env.action_size))
                else:
                    assert agent is not None
                    action = agent.act(observation, explore=False)

                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    if info["collision"]:
                        terminal_message = "COLLISION"
                    elif info["off_road"]:
                        terminal_message = "LEFT THE ROAD"
                    else:
                        terminal_message = "EPISODE COMPLETE"
                    terminal_frames = max(20, args.fps)

            renderer.render(
                env,
                policy_name=args.policy,
                action=action,
                episode=episode,
                episode_reward=episode_reward,
                epsilon=None if agent is None else 0.0,
                message=terminal_message,
            )
            renderer.tick()
    finally:
        renderer.close()


if __name__ == "__main__":
    main()

