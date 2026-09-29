"""Evaluate any policy on held-out worlds and print a reproducible table.

Training already reports per-difficulty numbers, but only for the agent it is
currently training. This module makes evaluation a first-class, standalone
thing: point it at any checkpoint — a DQN run, a behaviour clone, an old
snapshot — and score it against the same held-out worlds as everything else.

That matters for three reasons:

1. **BENCHMARK.md becomes reproducible.** The table is printed by a command
   anyone can run, rather than assembled by hand from a training log.
2. **Policies become comparable.** A clone trained by imitation and an agent
   trained by reinforcement can be graded on *identical* seeds, which is the
   only way the comparison means anything.
3. **Results stop going stale silently.** Re-run it after an environment
   change and the numbers move; a table nobody can regenerate just rots.

Every policy sees the same worlds. Seeds are derived from an explicit range,
never from a clock, so two runs of the same command produce the same table.

Examples
--------
    # One checkpoint across every difficulty cell
    python -m autodrive_rl.benchmark --policy models/autodrive_dqn_best.npz

    # The comparison: imitation vs reinforcement vs baselines, same worlds
    python -m autodrive_rl.benchmark \\
        --policy models/autodrive_dqn_best.npz \\
        --policy models/clone.npz \\
        --policy heuristic \\
        --policy random \\
        --episodes 100 --markdown

    # A single cell, more episodes, written to a file
    python -m autodrive_rl.benchmark --policy models/clone.npz \\
        --cells dense --episodes 200 --markdown --out BENCHMARK_dense.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Protocol, Sequence

import numpy as np

from .config import SCENARIO_PRESETS, EnvConfig, ScenarioSpec, resolve_scenario
from .dqn import DQNAgent
from .environment import DrivingEnv
from .heuristic import HeuristicDriver

DEFAULT_CELLS: tuple[str, ...] = ("sparse", "normal", "dense")

#: Held-out seed block. Training uses low seeds; these sit far away so a
#: benchmark world is never one the agent trained on.
DEFAULT_SEED_START = 350_000


# ── Policy adapters ──────────────────────────────────────────────────────────
# The three policy kinds have genuinely different interfaces: a DQN agent acts
# on the observation vector, the heuristic needs the environment object (it
# reads sensors and lane geometry directly), and a random policy needs neither.
# One tiny protocol lets a single rollout loop drive all of them, so no policy
# gets a subtly different evaluation than the others.


class Policy(Protocol):
    name: str

    #: Called once per episode with that episode's world seed. A policy with
    #: internal randomness must reseed from it: otherwise its action stream
    #: depends on how many episodes ran before, so the same cell scores
    #: differently depending on which *other* cells were requested alongside it.
    def reset(self, seed: int) -> None: ...

    def act(self, env: DrivingEnv, observation: np.ndarray) -> int: ...


class CheckpointPolicy:
    """A saved network — DQN agent or behaviour clone (same file format)."""

    def __init__(self, path: str | Path, *, seed: int) -> None:
        self.path = Path(path)
        self.name = self.path.stem
        self._agent = DQNAgent.load(self.path, seed=seed)

    def reset(self, seed: int) -> None:  # greedy, so stateless between episodes
        return None

    def act(self, env: DrivingEnv, observation: np.ndarray) -> int:
        # explore=False: greedy, so the number is the policy's, not epsilon's.
        return int(self._agent.act(observation, explore=False))


class HeuristicPolicy:
    """The hand-written rule-based driver."""

    name = "heuristic"

    def __init__(self) -> None:
        self._driver = HeuristicDriver()

    def reset(self, seed: int) -> None:
        self._driver.reset()

    def act(self, env: DrivingEnv, observation: np.ndarray) -> int:
        return int(self._driver.act(env))


class RandomPolicy:
    """Uniform random actions — the floor any real policy must clear."""

    name = "random"

    def __init__(self, *, seed: int, action_count: int = 5) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._action_count = action_count

    def reset(self, seed: int) -> None:
        # Derived from (base seed, episode seed) so episode N of a cell always
        # sees the same actions, whatever ran before it.
        self._rng = np.random.default_rng((self._seed, seed))

    def act(self, env: DrivingEnv, observation: np.ndarray) -> int:
        return int(self._rng.integers(self._action_count))


def build_policy(spec: str, *, seed: int) -> Policy:
    """`random`, `heuristic`, or a path to a .npz checkpoint."""
    if spec == "random":
        return RandomPolicy(seed=seed)
    if spec == "heuristic":
        return HeuristicPolicy()
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {path} "
            "(expected 'random', 'heuristic', or a path to a .npz file)"
        )
    return CheckpointPolicy(path, seed=seed)


# ── Rollouts ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CellResult:
    """One policy's performance on one difficulty cell."""

    policy: str
    cell: str
    episodes: int
    mean_return: float
    return_stdev: float
    mean_distance_m: float
    safe_rate: float
    collision_rate: float
    at_fault_rate: float
    stall_rate: float
    off_road_rate: float
    mean_speed_mps: float


def run_episode(
    policy: Policy,
    env_config: EnvConfig,
    scenario_spec: ScenarioSpec,
    *,
    seed: int,
) -> dict[str, float | bool]:
    """One deterministic episode. Same seed ⇒ same world, for every policy."""
    env = DrivingEnv(
        env_config,
        scenario="traffic",
        seed=seed,
        scenario_spec=scenario_spec,
    )
    observation, _ = env.reset(seed=seed)
    policy.reset(seed)

    episode_return = 0.0
    speeds: list[float] = []
    while True:
        action = policy.act(env, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        speeds.append(float(env.ego_speed_mps))
        if terminated or truncated:
            collision = bool(info["collision"])
            off_road = bool(info["off_road"])
            stalled = bool(info["stalled"])
            return {
                "return": episode_return,
                "distance_m": float(info["distance_m"]),
                "collision": collision,
                "at_fault": bool(info["at_fault"]),
                "stalled": stalled,
                "off_road": off_road,
                # Stalling in a live lane is a failure, not a safe outcome —
                # counting it as success is exactly how a safety score gets
                # won by not driving.
                "safe": not collision and not off_road and not stalled,
                "mean_speed_mps": fmean(speeds) if speeds else 0.0,
            }


def evaluate_cell(
    policy: Policy,
    env_config: EnvConfig,
    cell: str,
    *,
    episodes: int,
    seed_start: int,
) -> CellResult:
    spec = resolve_scenario(cell)
    episode_results = [
        run_episode(policy, env_config, spec, seed=seed_start + index)
        for index in range(episodes)
    ]
    returns = [float(r["return"]) for r in episode_results]
    return CellResult(
        policy=policy.name,
        cell=cell,
        episodes=episodes,
        mean_return=fmean(returns),
        # Population stdev: these episodes are the whole sample, not a draw
        # from a larger one. Reported so a reader can judge whether a gap
        # between two policies is real or noise.
        return_stdev=pstdev(returns) if len(returns) > 1 else 0.0,
        mean_distance_m=fmean(float(r["distance_m"]) for r in episode_results),
        safe_rate=sum(bool(r["safe"]) for r in episode_results) / episodes,
        collision_rate=sum(bool(r["collision"]) for r in episode_results) / episodes,
        at_fault_rate=sum(bool(r["at_fault"]) for r in episode_results) / episodes,
        stall_rate=sum(bool(r["stalled"]) for r in episode_results) / episodes,
        off_road_rate=sum(bool(r["off_road"]) for r in episode_results) / episodes,
        mean_speed_mps=fmean(float(r["mean_speed_mps"]) for r in episode_results),
    )


def benchmark(
    policy_specs: Sequence[str],
    *,
    cells: Sequence[str] = DEFAULT_CELLS,
    episodes: int = 100,
    seed_start: int = DEFAULT_SEED_START,
    max_steps: int = 900,
    policy_seed: int = 0,
    progress: bool = True,
) -> list[CellResult]:
    env_config = replace(EnvConfig(), max_steps=max_steps)
    results: list[CellResult] = []
    for spec in policy_specs:
        policy = build_policy(spec, seed=policy_seed)
        for cell in cells:
            if progress:
                print(f"  evaluating {policy.name} on {cell} …", file=sys.stderr)
            results.append(
                evaluate_cell(
                    policy,
                    env_config,
                    cell,
                    episodes=episodes,
                    seed_start=seed_start,
                )
            )
    return results


# ── Reporting ────────────────────────────────────────────────────────────────


def format_markdown(
    results: Iterable[CellResult],
    *,
    episodes: int,
    seed_start: int,
    max_steps: int,
) -> str:
    rows = list(results)
    cells = list(dict.fromkeys(r.cell for r in rows))
    seed_end = seed_start + episodes - 1

    lines = [
        "| Policy | Cell | Mean return | Safe completion | Collision | At fault | Stalled "
        "| Off-road | Distance | Mean speed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r.policy}` | {r.cell} "
            f"| {r.mean_return:,.1f} ± {r.return_stdev:,.0f} "
            f"| {r.safe_rate:.0%} "
            f"| {r.collision_rate:.0%} "
            f"| {r.at_fault_rate:.0%} "
            f"| {r.stall_rate:.0%} "
            f"| {r.off_road_rate:.0%} "
            f"| {r.mean_distance_m:,.0f} m "
            f"| {r.mean_speed_mps:.1f} m/s |"
        )

    preset_notes = ", ".join(
        _describe_cell(c) for c in cells if c in SCENARIO_PRESETS
    )

    lines += [
        "",
        f"{episodes} held-out episodes per cell, seeds `{seed_start}`–`{seed_end}`, "
        f"{max_steps}-step limit. Every policy is evaluated on the identical set "
        "of worlds, so differences between rows are differences in driving, not "
        "in luck.",
        "",
        f"Difficulty cells: {preset_notes}.",
        "",
        '"Safe completion" means the episode reached the step limit without a '
        "collision, an off-road event, or a stall. A stall is the ego sitting "
        "below the minimum speed in a live lane for longer than the grace "
        "period — blocking a motorway lane is a failure, not a safe outcome. "
        '"At fault" counts only the collisions the ego caused: running into '
        "something ahead, merging into someone, or being struck from behind "
        "while stopped. Return is shown as mean ± population standard "
        "deviation across episodes.",
        "",
        "Regenerate with:",
        "",
        "```bash",
        "python -m autodrive_rl.benchmark \\",
        *[f"  --policy {p} \\" for p in dict.fromkeys(r.policy for r in rows)],
        # Only name the cells when they differ from the default, so the common
        # case stays short — but never omit them when they would change the
        # result, which is how a table stops reproducing itself.
        *([f"  --cells {' '.join(cells)} \\"] if tuple(cells) != DEFAULT_CELLS else []),
        f"  --episodes {episodes} --seed-start {seed_start} --markdown",
        "```",
    ]
    return "\n".join(lines)


def _describe_cell(cell: str) -> str:
    """One phrase describing what a difficulty cell actually contains.

    Two cells that read identically here would make the table misleading —
    `unforgiving` shares its car and obstacle counts with `dense` — so every
    field that distinguishes a preset has to appear.
    """

    spec = SCENARIO_PRESETS[cell]
    parts = [
        f"{spec.traffic_count} cars",
        f"{spec.obstacle_count} obstacles",
        f"{spec.reactive_fraction:.0%} lane-changers",
    ]
    if spec.rear_traffic:
        parts.append("traffic from behind")
    if spec.inattentive_fraction > 0.0:
        parts.append(f"{spec.inattentive_fraction:.0%} inattentive drivers")
    return f"**{cell}** = " + " / ".join(parts)


def format_table(results: Iterable[CellResult]) -> str:
    rows = list(results)
    width = max((len(r.policy) for r in rows), default=6)
    lines = [
        f"{'policy'.ljust(width)}  {'cell':<7} {'return':>12} {'safe':>6} "
        f"{'coll':>6} {'off':>6} {'dist':>9} {'speed':>7}"
    ]
    lines.append("-" * len(lines[0]))
    for r in rows:
        lines.append(
            f"{r.policy.ljust(width)}  {r.cell:<7} "
            f"{r.mean_return:>8,.1f}±{r.return_stdev:<3.0f} "
            f"{r.safe_rate:>5.0%} {r.collision_rate:>5.0%} {r.off_road_rate:>5.0%} "
            f"{r.mean_distance_m:>7,.0f}m {r.mean_speed_mps:>5.1f}m/s"
        )
    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autodrive_rl.benchmark",
        description="Evaluate policies on held-out worlds and print a table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        metavar="SPEC",
        help="'random', 'heuristic', or a path to a .npz checkpoint. "
        "Repeat to compare several on identical worlds.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELLS),
        choices=sorted(SCENARIO_PRESETS),
        help="difficulty cells to evaluate (default: sparse normal dense)",
    )
    parser.add_argument("--episodes", type=int, default=100, help="episodes per cell")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=DEFAULT_SEED_START,
        help="first held-out world seed (default: %(default)s)",
    )
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="seed for the random policy; does not affect world generation",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="emit a markdown table suitable for pasting into BENCHMARK.md",
    )
    parser.add_argument("--out", type=Path, default=None, help="write output to a file")
    parser.add_argument("--quiet", action="store_true", help="suppress progress lines")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    policies = args.policies or ["random", "heuristic"]
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")

    results = benchmark(
        policies,
        cells=args.cells,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        policy_seed=args.policy_seed,
        progress=not args.quiet,
    )

    text = (
        format_markdown(
            results,
            episodes=args.episodes,
            seed_start=args.seed_start,
            max_steps=args.max_steps,
        )
        if args.markdown
        else format_table(results)
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
