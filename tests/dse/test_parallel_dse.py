# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""How DSE trials run in parallel: waves, constraint rejections, failed runs, cache and cancellation."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import CloudAIGymEnv, RewardOverrides, Runner
from cloudai.handlers import handle_dse_job

from .conftest import ProbeAgent
from .probe import (
    ScriptedRunner,
    ScriptedSystem,
    dse_runner,
    max_outstanding,
    probe_test_run,
    read_trajectory,
    waves,
)


def scripted(system: ScriptedSystem, x: list[int], max_parallelism: int, **kwargs) -> Runner:
    agent_config = {"max_parallelism": max_parallelism, **kwargs.pop("agent_config", {})}
    return dse_runner(system, ScriptedRunner, probe_test_run(x, agent_config=agent_config, **kwargs))


def probe_env(system: ScriptedSystem, x: list[int], max_parallelism: int, **kwargs) -> CloudAIGymEnv:
    test_run = probe_test_run(x, **kwargs)
    runner = dse_runner(system, ScriptedRunner, test_run)
    return CloudAIGymEnv(test_run, runner.runner, rewards=RewardOverrides(), max_parallelism=max_parallelism)


def test_grid_runs_in_waves(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3, 4, 5, 6, 7], max_parallelism=3)

    assert handle_dse_job(runner) == 0

    assert waves(system.events) == [[1, 2, 3], [4, 5, 6], [7]]
    assert max_outstanding(system.events) == 3
    assert read_trajectory(runner) == [(step, step, float(step)) for step in range(1, 8)]


def test_next_wave_starts_only_after_the_slowest_trial(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3, 4], max_parallelism=3, durations={2: 5})

    assert handle_dse_job(runner) == 0

    assert system.events.index(("finish", 2)) < system.events.index(("submit", 4))
    assert [step for event, step in system.events if event == "finish"][:3] == [1, 3, 2]


def test_trajectory_keeps_proposal_order_when_trials_finish_in_reverse(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3], max_parallelism=3, durations={1: 3, 2: 2, 3: 1})

    assert handle_dse_job(runner) == 0

    assert [event for event in system.events if event[0] == "finish"] == [("finish", 3), ("finish", 2), ("finish", 1)]
    assert read_trajectory(runner) == [(1, 1, 1.0), (2, 2, 2.0), (3, 3, 3.0)]


def test_trials_get_their_own_output_directory(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3], max_parallelism=3)

    assert handle_dse_job(runner) == 0

    trial_dirs = sorted(path.name for path in (runner.runner.scenario_root / "probe" / "0").iterdir() if path.is_dir())
    assert trial_dirs == ["1", "2", "3"]
    for step in (1, 2, 3):
        assert (runner.runner.scenario_root / "probe" / "0" / str(step) / "probe_result.txt").read_text() == str(step)


def test_rejected_proposals_take_no_slot_and_no_trial_number(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3, 4, 5, 6], max_parallelism=3, rejected_x=[2, 5])

    assert handle_dse_job(runner) == 0

    assert waves(system.events) == [[1, 2, 3], [4]]
    assert read_trajectory(runner) == [(1, 1, 1.0), (2, 3, 3.0), (3, 4, 4.0), (4, 6, 6.0)]


def test_failed_trial_does_not_stop_its_wave_or_the_sweep(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3, 4], max_parallelism=2, failing_x=[2])

    assert handle_dse_job(runner) == 0

    assert system.events == [
        ("submit", 1),
        ("submit", 2),
        ("finish", 1),
        ("fail", 2),
        ("submit", 3),
        ("submit", 4),
        ("finish", 3),
        ("finish", 4),
    ]
    assert read_trajectory(runner) == [(1, 1, 1.0), (2, 2, -1.0), (3, 3, 3.0), (4, 4, 4.0)]


@pytest.mark.parametrize("max_parallelism", [2, 3, 5, 16])
def test_trajectory_does_not_depend_on_parallelism(tmp_path: Path, max_parallelism: int) -> None:
    def sweep(root: Path, parallelism: int) -> tuple[list[tuple[int, int, float]], list[int]]:
        system = ScriptedSystem(name="scripted", install_path=root / "install", output_path=root / "output")
        runner = scripted(
            system,
            x=[1, 2, 3, 4, 5, 6, 1, 7, 3, 8],
            max_parallelism=parallelism,
            rejected_x=[2, 6],
            failing_x=[4],
            durations={1: 4, 3: 2, 5: 3},
        )
        assert handle_dse_job(runner) == 0
        assert max_outstanding(system.events) <= parallelism
        executed = sorted(step for event, step in system.events if event == "submit")
        return read_trajectory(runner), executed

    serial = sweep(tmp_path / "serial", 1)

    assert sweep(tmp_path / "parallel", max_parallelism) == serial
    trajectory, executed = serial
    assert [x for _, x, _ in trajectory] == [1, 3, 4, 5, 1, 7, 3, 8]
    assert executed == [1, 2, 3, 4, 6, 8]


def test_results_report_every_proposal_in_order(system: ScriptedSystem) -> None:
    env = probe_env(system, x=[1, 2, 3], max_parallelism=2, rejected_x=[2], failing_x=[3])

    results = env.evaluate_batch([(10, {"x": 1}), (11, {"x": 2}), (12, {"x": 3}), (13, {"x": 1})])

    assert [result["trial_index"] for result in results] == [10, 11, 12, 13]
    assert [result["status"] for result in results] == ["completed", "rejected", "failed", "cached"]
    assert [result["step"] for result in results] == [1, None, 2, 3]
    assert [result["value"] for result in results] == [1.0, -1.0, -1.0, 1.0]
    assert [result["observation"] for result in results] == [[1.0], [-1.0], [-1.0], [1.0]]
    assert [result["done"] for result in results] == [False, True, False, False]
    trial_root = env.runner.scenario_root / "probe" / "0"
    assert [result["output_path"] for result in results] == [trial_root / "1", None, trial_root / "2", trial_root / "1"]
    assert env.test_run.step == 3
    assert env.upcoming_trial == 4


def test_earlier_waves_feed_the_cache_of_later_ones(system: ScriptedSystem) -> None:
    env = probe_env(system, x=[1, 2, 3], max_parallelism=2)

    results = env.evaluate_batch(enumerate([{"x": 1}, {"x": 2}, {"x": 1}, {"x": 3}, {"x": 2}], start=1))

    assert [result["status"] for result in results] == ["completed", "completed", "cached", "completed", "cached"]
    assert waves(system.events) == [[1, 2], [4]]


def test_identical_proposals_in_one_wave_run_once(system: ScriptedSystem) -> None:
    env = probe_env(system, x=[1, 2], max_parallelism=3, failing_x=[2])

    results = env.evaluate_batch(enumerate([{"x": 1}, {"x": 1}, {"x": 2}, {"x": 2}, {"x": 1}], start=1))

    assert [result["status"] for result in results] == ["completed", "cached", "failed", "cached", "cached"]
    assert [result["value"] for result in results] == [1.0, 1.0, -1.0, -1.0, 1.0]
    assert [result["step"] for result in results] == [1, 2, 3, 4, 5]
    trial_root = env.runner.scenario_root / "probe" / "0"
    assert [result["output_path"] for result in results] == [trial_root / str(step) for step in (1, 1, 3, 3, 1)]
    assert waves(system.events) == [[1, 3]]


def test_proposals_are_pulled_lazily(system: ScriptedSystem) -> None:
    env = probe_env(system, x=[1, 2, 3, 4, 5], max_parallelism=2)
    pulled: list[tuple[int, list[tuple[str, int]]]] = []

    def proposals():
        for x in (1, 2, 3, 4, 5):
            pulled.append((x, list(system.events)))
            yield x, {"x": x}

    env.evaluate_batch(proposals())

    assert [seen for _, seen in pulled][:2] == [[], []]
    assert ("finish", 2) in pulled[2][1]
    assert ("finish", 4) in pulled[4][1]


def test_step_allowance_bounds_an_endless_stream(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3], max_parallelism=2, agent="probe_stream", agent_steps=5, rejected_x=[2])

    assert handle_dse_job(runner) == 0

    assert waves(system.events) == [[1, 2]]
    assert read_trajectory(runner) == [(1, 1, 1.0), (2, 3, 3.0), (3, 1, 1.0), (4, 3, 3.0), (5, 1, 1.0)]
    [agent] = ProbeAgent.created
    assert [update["status"] for update in agent.updates] == [
        "completed",
        "rejected",
        "completed",
        "cached",
        "rejected",
        "cached",
        "cached",
    ]


def test_stream_of_rejected_proposals_ends(system: ScriptedSystem) -> None:
    runner = scripted(
        system,
        x=[1, 2],
        max_parallelism=4,
        agent="probe_stream",
        agent_steps=2,
        rejected_x=[1, 2],
        agent_config={"max_consecutive_rejected_proposals": 3},
    )

    assert handle_dse_job(runner) == 0

    assert system.events == []
    assert read_trajectory(runner) == []
    [agent] = ProbeAgent.created
    assert [update["status"] for update in agent.updates] == ["rejected"] * 6


def test_skipped_step_consumes_a_trial_number(system: ScriptedSystem) -> None:
    env = probe_env(system, x=[1, 2, 3], max_parallelism=2, rejected_x=[2])
    env.max_consecutive_rejected_proposals = 2

    results = env.evaluate_batch(enumerate([{"x": 1}, {"x": 2}, {"x": 2}, {"x": 3}], start=1), max_steps=3)

    assert [result["status"] for result in results] == ["completed", "rejected", "rejected", "completed"]
    assert [result["step"] for result in results] == [1, None, None, 3]
    assert env.evaluate_batch([(5, {"x": 1})], max_steps=3) == []


def test_serial_agent_stays_serial(system: ScriptedSystem, caplog: pytest.LogCaptureFixture) -> None:
    runner = scripted(system, x=[1, 2, 3], max_parallelism=4, agent="probe_serial", agent_steps=3, rejected_x=[2])

    with caplog.at_level(logging.WARNING):
        assert handle_dse_job(runner) == 0

    assert max_outstanding(system.events) == 1
    assert "max_parallelism=4 has no effect" in caplog.text
    [agent] = ProbeAgent.created
    assert agent.log == [
        ("propose", 1),
        ("update", 1),
        ("propose", 2),
        ("update", 2),
        ("propose", 3),
        ("update", 3),
        ("propose", 4),
        ("update", 4),
    ]
    assert [update["status"] for update in agent.updates] == ["completed", "rejected", "completed", "cached"]


def test_agent_gets_feedback_for_a_whole_batch_before_the_next_one(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3, 4, 5, 6], max_parallelism=2, agent="probe_generation", agent_steps=6)

    assert handle_dse_job(runner) == 0

    assert waves(system.events) == [[1, 2], [3], [4, 5], [6]]
    [agent] = ProbeAgent.created
    proposes = [("propose", i) for i in range(1, 7)]
    updates = [("update", i) for i in range(1, 7)]
    assert agent.log == [*proposes[:3], *updates[:3], *proposes[3:], *updates[3:]]
    assert all(update["prev_observation"] == [0.0] for update in agent.updates[:3])
    assert all(update["prev_observation"] == [3.0] for update in agent.updates[3:])


def test_env_params_are_sampled_per_trial_whatever_the_parallelism(tmp_path: Path) -> None:
    def sweep(root: Path, parallelism: int) -> list[dict]:
        system = ScriptedSystem(name="scripted", install_path=root / "install", output_path=root / "output")
        test_run = probe_test_run([1, 2, 3], agent="probe_stream", agent_steps=6, rejected_x=[2])
        test_run.test.cmd_args.noise = [10, 20, 30, 40]
        test_run.test.env_params = {"noise": EnvParamSpec()}
        runner = dse_runner(system, ScriptedRunner, test_run)
        env = CloudAIGymEnv(test_run, runner.runner, rewards=RewardOverrides(), max_parallelism=parallelism)
        results = env.evaluate_batch(enumerate([{"x": x} for x in (1, 2, 3, 1, 3, 2, 1)], start=1))
        return [{"step": r["step"], "status": r["status"], "env_params": r["env_params"]} for r in results]

    serial = sweep(tmp_path / "serial", 1)

    assert sweep(tmp_path / "parallel", 3) == serial
    assert len({result["env_params"]["noise"] for result in serial}) > 1


def test_step_is_a_one_proposal_batch(system: ScriptedSystem) -> None:
    env = probe_env(system, x=[1, 2], max_parallelism=3, rejected_x=[2])

    assert env.step({"x": 1}) == ([1.0], 1.0, False, {})
    assert env.step({"x": 2}) == ([-1.0], -1.0, True, {})
    assert env.step({"x": 1}) == ([1.0], 1.0, False, {})
    assert waves(system.events) == [[1]]


def test_cancelling_stops_the_sweep_and_drops_the_interrupted_wave(system: ScriptedSystem) -> None:
    first = probe_test_run([1, 2, 3, 4, 5, 6], agent_config={"max_parallelism": 2}, duration=3)
    second = probe_test_run([1, 2], name="second")
    runner = dse_runner(system, ScriptedRunner, first, second)

    def cancel_during_second_wave() -> None:
        if ("submit", 4) in system.events and not runner.runner.shutting_down:
            runner.runner.shutdown()

    system.on_pass(cancel_during_second_wave)

    assert handle_dse_job(runner) == 1

    assert system.events == [
        ("submit", 1),
        ("submit", 2),
        ("finish", 1),
        ("finish", 2),
        ("submit", 3),
        ("submit", 4),
        ("kill", 3),
        ("kill", 4),
    ]
    assert read_trajectory(runner) == [(1, 1, 1.0), (2, 2, 2.0)]
    assert read_trajectory(runner, "second") == []
    assert runner.runner.jobs == []


def test_step_fails_once_the_run_is_cancelled(system: ScriptedSystem) -> None:
    env = probe_env(system, x=[1, 2], max_parallelism=1)
    env.runner.shutdown()

    with pytest.raises(RuntimeError, match="shutting down"):
        env.step({"x": 1})

    assert system.events == []


def test_submission_error_stops_the_wave_and_fails_the_sweep(system: ScriptedSystem) -> None:
    runner = scripted(system, x=[1, 2, 3, 4], max_parallelism=3, unsubmittable_x=[3], duration=2)

    assert handle_dse_job(runner) == 1

    assert system.events == [("submit", 1), ("submit", 2), ("kill", 1), ("kill", 2)]
    assert read_trajectory(runner) == []


def test_pin_nodes_cannot_be_combined_with_parallel_trials(
    system: ScriptedSystem, caplog: pytest.LogCaptureFixture
) -> None:
    test_run = probe_test_run([1, 2], agent_config={"max_parallelism": 2})
    test_run.pin_nodes = True
    runner = dse_runner(system, ScriptedRunner, test_run)

    assert handle_dse_job(runner) == 1

    assert system.events == []
    assert "'pin_nodes' is not supported with 'max_parallelism' greater than 1" in caplog.text
