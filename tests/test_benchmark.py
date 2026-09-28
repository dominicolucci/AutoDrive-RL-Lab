"""Tests for the standalone benchmark harness.

The important property is **fairness**: every policy must be scored on the
identical set of worlds. If seeding drifted between policies, a comparison
table would silently measure luck instead of driving, and every conclusion
drawn from it would be wrong. Those tests come first.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from autodrive_rl.benchmark import (
    DEFAULT_CELLS,
    CheckpointPolicy,
    HeuristicPolicy,
    RandomPolicy,
    benchmark,
    build_parser,
    build_policy,
    evaluate_cell,
    format_markdown,
    format_table,
    run_episode,
)
from autodrive_rl.config import EnvConfig, resolve_scenario

SHORT = replace(EnvConfig(), max_steps=40)


# ── Fairness: identical worlds across policies ───────────────────────────────


def test_same_seed_gives_the_same_world_to_different_policies():
    """The whole comparison rests on this."""
    spec = resolve_scenario("normal")
    from autodrive_rl.environment import DrivingEnv

    first = DrivingEnv(SHORT, scenario="traffic", seed=4242, scenario_spec=spec)
    obs_a, _ = first.reset(seed=4242)
    second = DrivingEnv(SHORT, scenario="traffic", seed=4242, scenario_spec=spec)
    obs_b, _ = second.reset(seed=4242)

    np.testing.assert_allclose(obs_a, obs_b)


def test_one_policy_is_reproducible_across_runs():
    spec = resolve_scenario("normal")
    policy = HeuristicPolicy()
    first = run_episode(policy, SHORT, spec, seed=99)
    second = run_episode(policy, SHORT, spec, seed=99)
    assert first["return"] == pytest.approx(second["return"])
    assert first["distance_m"] == pytest.approx(second["distance_m"])


def test_different_seeds_give_different_worlds():
    """Held-out seeds must actually generate distinct traffic.

    Asserted on the starting observation rather than the episode outcome: two
    different worlds can still produce identical short-horizon results (the
    heuristic simply accelerates while distant traffic is irrelevant), so the
    outcome is a lossy proxy for the property we care about.
    """
    from autodrive_rl.environment import DrivingEnv

    spec = resolve_scenario("normal")
    starts = []
    for seed in (1, 2, 3):
        env = DrivingEnv(SHORT, scenario="traffic", seed=seed, scenario_spec=spec)
        observation, _ = env.reset(seed=seed)
        starts.append(observation.copy())

    assert not np.allclose(starts[0], starts[1])
    assert not np.allclose(starts[0], starts[2])


def test_seed_range_produces_varied_outcomes_over_a_full_episode():
    """Over a realistic horizon, different worlds do diverge in outcome."""
    spec = resolve_scenario("normal")
    policy = HeuristicPolicy()
    horizon = replace(EnvConfig(), max_steps=300)
    distances = {
        round(run_episode(policy, horizon, spec, seed=seed)["distance_m"], 3)
        for seed in range(4)
    }
    assert len(distances) > 1, "held-out worlds should not all play out identically"


# ── Episode bookkeeping ──────────────────────────────────────────────────────


def test_episode_result_has_the_expected_shape():
    result = run_episode(HeuristicPolicy(), SHORT, resolve_scenario("sparse"), seed=7)
    assert set(result) == {
        "return",
        "distance_m",
        "collision",
        "off_road",
        "safe",
        "mean_speed_mps",
    }
    assert result["distance_m"] >= 0.0
    assert result["mean_speed_mps"] >= 0.0


def test_safe_means_neither_collision_nor_off_road():
    for seed in range(4):
        r = run_episode(HeuristicPolicy(), SHORT, resolve_scenario("sparse"), seed=seed)
        assert r["safe"] == (not r["collision"] and not r["off_road"])


def test_rates_are_fractions_and_consistent():
    result = evaluate_cell(
        HeuristicPolicy(), SHORT, "sparse", episodes=4, seed_start=500
    )
    assert result.episodes == 4
    for rate in (result.safe_rate, result.collision_rate, result.off_road_rate):
        assert 0.0 <= rate <= 1.0
    # An episode is safe, or it collided, or it left the road — a policy cannot
    # be safe more often than the failures leave room for.
    assert result.safe_rate <= 1.0 - max(result.collision_rate, result.off_road_rate) + 1e-9


def test_stdev_is_zero_for_a_single_episode():
    result = evaluate_cell(
        HeuristicPolicy(), SHORT, "sparse", episodes=1, seed_start=1
    )
    assert result.return_stdev == 0.0


# ── Policy adapters ──────────────────────────────────────────────────────────


def test_random_policy_returns_valid_actions():
    policy = RandomPolicy(seed=3)
    actions = {policy.act(None, np.zeros(16)) for _ in range(60)}  # type: ignore[arg-type]
    assert actions <= {0, 1, 2, 3, 4}
    assert len(actions) > 1, "a random policy should not emit a constant action"


def test_random_policy_is_seeded():
    a = [RandomPolicy(seed=11).act(None, np.zeros(16)) for _ in range(5)]  # type: ignore[arg-type]
    b = [RandomPolicy(seed=11).act(None, np.zeros(16)) for _ in range(5)]  # type: ignore[arg-type]
    assert a == b


def test_build_policy_resolves_the_named_baselines():
    assert isinstance(build_policy("random", seed=0), RandomPolicy)
    assert isinstance(build_policy("heuristic", seed=0), HeuristicPolicy)


def test_build_policy_rejects_a_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        build_policy("models/definitely-not-here.npz", seed=0)


@pytest.mark.skipif(
    not Path("models/autodrive_dqn_best.npz").exists(),
    reason="bundled checkpoint not present",
)
def test_checkpoint_policy_loads_and_acts():
    policy = CheckpointPolicy("models/autodrive_dqn_best.npz", seed=0)
    assert policy.name == "autodrive_dqn_best"
    action = policy.act(None, np.zeros(16, dtype=np.float32))  # type: ignore[arg-type]
    assert action in {0, 1, 2, 3, 4}


@pytest.mark.skipif(
    not Path("models/autodrive_dqn_best.npz").exists(),
    reason="bundled checkpoint not present",
)
def test_checkpoint_policy_is_greedy_and_deterministic():
    """Evaluation must report the policy, not the exploration noise."""
    policy = CheckpointPolicy("models/autodrive_dqn_best.npz", seed=0)
    observation = np.linspace(-1.0, 1.0, 16).astype(np.float32)
    assert len({policy.act(None, observation) for _ in range(20)}) == 1  # type: ignore[arg-type]


# ── Orchestration and reporting ──────────────────────────────────────────────


def test_benchmark_covers_every_policy_and_cell():
    results = benchmark(
        ["random", "heuristic"],
        cells=("sparse", "normal"),
        episodes=1,
        seed_start=900,
        max_steps=30,
        progress=False,
    )
    assert len(results) == 4
    assert {r.policy for r in results} == {"random", "heuristic"}
    assert {r.cell for r in results} == {"sparse", "normal"}


def test_markdown_reports_seeds_and_a_reproduction_command():
    results = benchmark(
        ["random"], cells=("sparse",), episodes=2, seed_start=350_000,
        max_steps=30, progress=False,
    )
    text = format_markdown(results, episodes=2, seed_start=350_000, max_steps=30)
    assert "| Policy | Cell |" in text
    assert "350000" in text and "350001" in text, "seed range must be stated"
    assert "python -m autodrive_rl.benchmark" in text, "table must be reproducible"
    assert "identical set" in text


def test_a_cell_scores_the_same_whatever_ran_alongside_it():
    """The random baseline used to carry one RNG stream across every episode of
    every cell, so asking for `dense` alone and asking for `sparse dense` gave
    different `dense` rows. A benchmark whose answer depends on the question is
    not a benchmark."""
    alone = benchmark(
        ["random"], cells=("dense",), episodes=3, seed_start=350_000,
        max_steps=60, progress=False,
    )
    alongside = benchmark(
        ["random"], cells=("sparse", "dense"), episodes=3, seed_start=350_000,
        max_steps=60, progress=False,
    )
    dense_alone = next(r for r in alone if r.cell == "dense")
    dense_alongside = next(r for r in alongside if r.cell == "dense")
    assert dense_alone == dense_alongside


def test_markdown_names_non_default_cells_in_the_reproduction_command():
    """Omitting `--cells` when the run used a non-default set would print a
    command that regenerates a *different* table."""
    results = benchmark(
        ["random"], cells=("unforgiving",), episodes=1, seed_start=350_000,
        max_steps=30, progress=False,
    )
    text = format_markdown(results, episodes=1, seed_start=350_000, max_steps=30)
    assert "--cells unforgiving" in text


def test_markdown_distinguishes_cells_that_differ_only_in_the_new_fields():
    """`unforgiving` shares its car and obstacle counts with `dense`; a legend
    printing only those would describe two different worlds identically."""
    results = benchmark(
        ["random"], cells=("dense", "unforgiving"), episodes=1, seed_start=350_000,
        max_steps=30, progress=False,
    )
    text = format_markdown(results, episodes=1, seed_start=350_000, max_steps=30)
    assert "traffic from behind" in text
    assert "inattentive" in text


def test_plain_table_renders():
    results = benchmark(
        ["random"], cells=("sparse",), episodes=1, seed_start=1,
        max_steps=30, progress=False,
    )
    assert "policy" in format_table(results)


def test_default_cells_match_the_configured_presets():
    from autodrive_rl.config import SCENARIO_PRESETS

    assert set(DEFAULT_CELLS) <= set(SCENARIO_PRESETS)


def test_cli_defaults_to_the_baseline_policies():
    args = build_parser().parse_args([])
    assert args.policies is None  # main() substitutes random + heuristic
    assert args.cells == list(DEFAULT_CELLS)
    assert args.episodes == 100
    assert args.seed_start == 350_000


def test_cli_accepts_repeated_policies():
    args = build_parser().parse_args(
        ["--policy", "random", "--policy", "models/clone.npz", "--markdown"]
    )
    assert args.policies == ["random", "models/clone.npz"]
    assert args.markdown is True
