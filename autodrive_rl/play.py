"""Run the visual simulator with manual, heuristic, random, or DQN control."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .clone import DemoRecorder
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
    parser.add_argument(
        "--autopilot-model",
        type=Path,
        default=Path("models/autopilot.npz"),
        help=(
            "saved .npz model engaged by SPACE while driving manually; train "
            "one with `python -m autodrive_rl.train --handover "
            "--output models/autopilot.npz`"
        ),
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        help=(
            "record your manual driving to this .npz for behavior cloning "
            "(crashed episodes and autopilot frames are excluded); train a "
            "clone with `python -m autodrive_rl.clone --demos <file>`"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.record is not None and args.policy != "manual":
        raise SystemExit("--record captures YOUR driving; use it with --policy manual")
    recorder = DemoRecorder() if args.record is not None else None
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

    # Autopilot for manual driving: engaged/disengaged with SPACE, and any
    # steering/brake/gas press hands control straight back to the driver.
    autopilot_agent: DQNAgent | None = None
    if args.policy == "manual" and args.autopilot_model.exists():
        autopilot_agent = DQNAgent.load(args.autopilot_model, seed=args.seed)
        if (
            autopilot_agent.observation_size != env.observation_size
            or autopilot_agent.action_size != env.action_size
        ):
            print("Autopilot model does not match this environment version; ignoring.")
            autopilot_agent = None
    autopilot_on = False
    autopilot_notice = 0

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
                if recorder is not None:
                    recorder.end_episode(crashed=False)
                observation, _ = env.reset(seed=args.seed + episode)
                heuristic.reset()
                episode_reward = 0.0
                terminal_message = None
                terminal_frames = 0
                autopilot_on = False

            if renderer.autopilot_toggle_requested:
                renderer.autopilot_toggle_requested = False
                if args.policy == "manual":
                    if autopilot_agent is None:
                        autopilot_notice = 2 * args.fps
                    else:
                        autopilot_on = not autopilot_on
            if autopilot_notice > 0:
                autopilot_notice -= 1

            if terminal_frames > 0:
                terminal_frames -= 1
                if terminal_frames == 0:
                    episode += 1
                    observation, _ = env.reset(seed=args.seed + episode)
                    heuristic.reset()
                    episode_reward = 0.0
                    terminal_message = None
                    autopilot_on = False
            elif not renderer.paused:
                if args.policy == "manual":
                    manual = renderer.manual_action()
                    if autopilot_on and manual != int(Action.MAINTAIN):
                        autopilot_on = False  # instant driver takeover
                    if autopilot_on and autopilot_agent is not None:
                        action = autopilot_agent.act(observation, explore=False)
                    else:
                        action = manual
                elif args.policy == "heuristic":
                    action = heuristic.act(env)
                elif args.policy == "random":
                    action = int(env.rng.integers(0, env.action_size))
                else:
                    assert agent is not None
                    action = agent.act(observation, explore=False)

                observation_before = observation
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                if recorder is not None and not autopilot_on:
                    recorder.add(observation_before, action)
                if terminated or truncated:
                    if recorder is not None:
                        recorder.end_episode(
                            crashed=bool(info["collision"] or info["off_road"])
                        )
                    if info["collision"]:
                        terminal_message = "COLLISION"
                    elif info["off_road"]:
                        terminal_message = "LEFT THE ROAD"
                    else:
                        terminal_message = (
                            f"EPISODE COMPLETE  -  GRADE {info['grade_letter']}"
                        )
                    terminal_frames = max(20, args.fps)

            autopilot_state: str | None = None
            if args.policy == "manual":
                if autopilot_notice > 0:
                    autopilot_state = "missing"
                elif autopilot_on:
                    autopilot_state = "on"
                else:
                    autopilot_state = "off"
            q_values = None
            if args.policy == "dqn" and agent is not None:
                q_values = agent.q_values(observation)
            elif autopilot_on and autopilot_agent is not None:
                q_values = autopilot_agent.q_values(observation)
            renderer.render(
                env,
                policy_name=args.policy,
                action=action,
                episode=episode,
                episode_reward=episode_reward,
                epsilon=None if agent is None else 0.0,
                message=terminal_message,
                autopilot=autopilot_state,
                q_values=q_values,
            )
            renderer.tick()
    finally:
        renderer.close()
        if recorder is not None:
            try:
                saved = recorder.save(args.record)
            except ValueError as error:
                print(f"Nothing recorded: {error}")
            else:
                print(
                    f"Recorded {recorder.kept_steps} steps of safe driving to "
                    f"{saved} ({recorder.dropped_episodes} crashed episode(s) "
                    "dropped)."
                )
                print(
                    "Train your clone:  python -m autodrive_rl.clone "
                    f"--demos {saved} --output models/clone.npz"
                )


if __name__ == "__main__":
    main()