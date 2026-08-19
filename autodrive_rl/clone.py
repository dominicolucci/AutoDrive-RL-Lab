"""Behavior cloning: learn to drive purely from recorded human demonstrations.

This is the supervised-learning half of the paradigm map. Where the DQN
learns from reward (nobody shows it how to drive — it finds out), the clone
learns from imitation: it never sees a reward, only `(observation, action)`
pairs recorded while a human drove, and it minimizes cross-entropy between
its action distribution and what the human actually pressed.

The clone uses the exact same 16 -> hidden -> 5 network as the DQN and is
saved in the same checkpoint format, so a trained clone is a drop-in policy:
play it with `--policy dqn --model models/clone.npz`, engage it as the
autopilot, or grade it against the RL agent on identical worlds.

Workflow:

    # 1. Drive and record (crashed episodes are dropped automatically)
    python -m autodrive_rl.play --policy manual --record demos/me.npz

    # 2. Clone the driving
    python -m autodrive_rl.clone --demos demos/me.npz --output models/clone.npz

    # 3. Watch yourself drive
    python -m autodrive_rl.play --policy dqn --model models/clone.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dqn import MLP, Adam


class DemoRecorder:
    """Collect `(observation, action)` pairs from a human driving session.

    Episodes that end in a collision or off-road are dropped on save — the
    point of imitation is to clone the driving worth imitating.
    """

    def __init__(self) -> None:
        self._episodes: list[tuple[list[np.ndarray], list[int]]] = []
        self._observations: list[np.ndarray] = []
        self._actions: list[int] = []
        self.dropped_episodes = 0

    def add(self, observation: np.ndarray, action: int) -> None:
        self._observations.append(np.asarray(observation, dtype=np.float32).copy())
        self._actions.append(int(action))

    def end_episode(self, *, crashed: bool) -> None:
        if crashed:
            self.dropped_episodes += 1
        elif self._observations:
            self._episodes.append((self._observations, self._actions))
        self._observations = []
        self._actions = []

    @property
    def kept_steps(self) -> int:
        return sum(len(actions) for _, actions in self._episodes) + len(self._actions)

    def save(self, path: str | Path) -> Path:
        """Write kept episodes (plus any safe in-progress driving) to .npz."""

        episodes = list(self._episodes)
        if self._observations:  # session ended mid-episode without a crash
            episodes.append((self._observations, self._actions))
        if not episodes:
            raise ValueError("no safe driving recorded - nothing to save")
        observations = np.concatenate(
            [np.stack(obs) for obs, _ in episodes]
        ).astype(np.float32)
        actions = np.concatenate(
            [np.asarray(acts, dtype=np.int64) for _, acts in episodes]
        )
        episode_ids = np.concatenate(
            [np.full(len(acts), index, dtype=np.int64) for index, (_, acts) in enumerate(episodes)]
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            observations=observations,
            actions=actions,
            episode_ids=episode_ids,
        )
        return path


def load_demos(paths: Sequence[str | Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate one or more recorded demo files."""

    all_observations: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    for path in paths:
        with np.load(Path(path), allow_pickle=False) as data:
            all_observations.append(np.asarray(data["observations"], dtype=np.float32))
            all_actions.append(np.asarray(data["actions"], dtype=np.int64))
    return np.concatenate(all_observations), np.concatenate(all_actions)


@dataclass(frozen=True)
class CloneResult:
    train_accuracy: float
    validation_accuracy: float
    final_loss: float
    epochs: int
    samples: int


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_clone(
    observations: np.ndarray,
    actions: np.ndarray,
    *,
    action_size: int = 5,
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 60,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    seed: int = 0,
    log_every: int = 10,
    verbose: bool = True,
) -> tuple[MLP, CloneResult]:
    """Supervised training: minimize cross-entropy against the human actions."""

    observations = np.asarray(observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int64)
    if len(observations) != len(actions):
        raise ValueError("observations and actions must be the same length")
    if len(observations) < 10:
        raise ValueError("need at least 10 demonstration steps")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(observations))
    observations, actions = observations[order], actions[order]
    validation_count = max(1, int(len(observations) * validation_fraction))
    val_obs, val_actions = observations[:validation_count], actions[:validation_count]
    train_obs, train_actions = observations[validation_count:], actions[validation_count:]

    network = MLP(observations.shape[1], action_size, hidden_sizes, seed=seed)
    parameters: list[np.ndarray] = []
    for weight, bias in zip(network.weights, network.biases):
        parameters.extend([weight, bias])
    optimizer = Adam(parameters, learning_rate)

    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        epoch_order = rng.permutation(len(train_obs))
        losses: list[float] = []
        for start in range(0, len(epoch_order), batch_size):
            batch = epoch_order[start : start + batch_size]
            logits, cache = network.forward(train_obs[batch], cache=True)
            probabilities = _softmax(np.asarray(logits))
            batch_actions = train_actions[batch]
            picked = probabilities[np.arange(len(batch)), batch_actions]
            losses.append(float(-np.log(np.clip(picked, 1e-12, None)).mean()))
            gradient = probabilities.copy()
            gradient[np.arange(len(batch)), batch_actions] -= 1.0
            gradient /= len(batch)
            grad_weights, grad_biases = network.backward(cache, gradient)
            interleaved: list[np.ndarray] = []
            for grad_weight, grad_bias in zip(grad_weights, grad_biases):
                interleaved.extend([grad_weight, grad_bias])
            optimizer.step(interleaved)
        final_loss = float(np.mean(losses))
        if verbose and (epoch % max(1, log_every) == 0 or epoch in (1, epochs)):
            train_acc = _accuracy(network, train_obs, train_actions)
            val_acc = _accuracy(network, val_obs, val_actions)
            print(
                f"epoch {epoch:3d}/{epochs}  loss={final_loss:.4f}  "
                f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}"
            )

    result = CloneResult(
        train_accuracy=_accuracy(network, train_obs, train_actions),
        validation_accuracy=_accuracy(network, val_obs, val_actions),
        final_loss=final_loss,
        epochs=epochs,
        samples=len(observations),
    )
    return network, result


def _accuracy(network: MLP, observations: np.ndarray, actions: np.ndarray) -> float:
    if len(observations) == 0:
        return float("nan")
    logits = np.asarray(network.forward(observations))
    return float((np.argmax(logits, axis=1) == actions).mean())


def save_clone(
    network: MLP, path: str | Path, *, observation_size: int, action_size: int
) -> Path:
    """Write a DQNAgent-compatible checkpoint so the clone is a drop-in policy."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hidden_sizes = [weight.shape[0] for weight in network.weights[1:]]
    payload: dict[str, np.ndarray] = {
        "observation_size": np.array(observation_size, dtype=np.int64),
        "action_size": np.array(action_size, dtype=np.int64),
        "hidden_sizes": np.asarray(hidden_sizes, dtype=np.int64),
        "environment_steps": np.array(0, dtype=np.int64),
    }
    for index, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        payload[f"weight_{index}"] = weight
        payload[f"bias_{index}"] = bias
    np.savez_compressed(path, **payload)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demos",
        type=Path,
        nargs="+",
        required=True,
        help="one or more .npz files recorded with `play --record`",
    )
    parser.add_argument("--output", type=Path, default=Path("models/clone.npz"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate the clone on the scenario presets after training",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    observations, actions = load_demos(args.demos)
    action_counts = np.bincount(actions, minlength=5)
    print(
        f"loaded {len(actions)} steps from {len(args.demos)} file(s)  "
        f"action mix: maintain={action_counts[0]} accel={action_counts[1]} "
        f"brake={action_counts[2]} left={action_counts[3]} right={action_counts[4]}"
    )
    network, result = train_clone(
        observations,
        actions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.val_fraction,
        seed=args.seed,
    )
    save_clone(
        network,
        args.output,
        observation_size=observations.shape[1],
        action_size=5,
    )
    print(
        f"\nSaved clone: {args.output}  "
        f"(train_acc={result.train_accuracy:.3f}, val_acc={result.validation_accuracy:.3f})"
    )
    print(
        "Drive it with:  python -m autodrive_rl.play --policy dqn "
        f"--model {args.output}"
    )
    if args.eval:
        from .config import SCENARIO_PRESETS, EnvConfig
        from .dqn import DQNAgent
        from .train import evaluate

        agent = DQNAgent.load(args.output, seed=args.seed)
        print("\nClone report card on held-out worlds:")
        for preset_name, spec in SCENARIO_PRESETS.items():
            scores: dict[str, Any] = evaluate(
                agent,
                EnvConfig(),
                episodes=3,
                seed=args.seed + 424_242,
                scenario="traffic",
                scenario_spec=spec,
            )
            print(
                f"  {preset_name:7s}  return={scores['return']:8.1f}  "
                f"distance={scores['distance_m']:7.1f} m  "
                f"safe_rate={scores['safe_rate']:.2f}"
            )


if __name__ == "__main__":
    main()