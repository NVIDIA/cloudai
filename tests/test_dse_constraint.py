# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
import toml
from pydantic import ValidationError

from cloudai.core import ConstraintEvaluationError, DSEConstraints, TestRun
from cloudai.workloads.sleep import SleepCmdArgs, SleepTestDefinition


def test_constraints_share_bound_variables_and_report_first_failure() -> None:
    constraints = DSEConstraints(
        variables={
            "tp": "cmd_args.tensor_parallel_size",
            "pp": "cmd_args.pipeline_parallel_size",
            "gpus_per_node": "system.gpus_per_node",
            "mode": "extra_env_vars.MODE",
        },
        expressions={
            "parallelism_fits": "tp * pp <= gpus_per_node",
            "mode_supported": "mode in ['fast', 'safe']",
        },
    )
    context = {
        "cmd_args": {"tensor_parallel_size": 4, "pipeline_parallel_size": 2},
        "extra_env_vars": {"MODE": "fast"},
        "system": {"gpus_per_node": 8},
        "test_run": {"num_nodes": 1},
    }

    assert constraints.evaluate(context) == (True, None, None)

    context["cmd_args"]["pipeline_parallel_size"] = 4
    assert constraints.evaluate(context) == (False, "parallelism_fits", "tp * pp <= gpus_per_node")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "cmd_args.__class__",
        "sum([1, 2]) <= 3",
    ],
)
def test_constraints_reject_unsafe_expressions(expression: str) -> None:
    with pytest.raises(ValidationError, match=r"does not allow|private attributes"):
        DSEConstraints(expressions={"unsafe": expression})


def test_constraints_reject_unknown_variable_at_parse_time() -> None:
    with pytest.raises(ValidationError, match="Constraint 'unknown' uses unknown variables: tp"):
        DSEConstraints(expressions={"unknown": "tp <= 8"})


@pytest.mark.parametrize(
    "path",
    [
        "cmd_args",
        "unknown.value",
        "cmd_args.",
        "cmd_args..tp",
        "cmd_args._private",
    ],
)
def test_constraints_reject_invalid_variable_paths(path: str) -> None:
    with pytest.raises(ValidationError, match="Constraint variable path"):
        DSEConstraints(variables={"value": path}, expressions={"valid": "value == 1"})


@pytest.mark.parametrize(
    "expression",
    [
        "enabled and not disabled",
        "mode == 'fast' or test_run.num_nodes == 1",
        "0 < tp <= system.gpus_per_node",
        "tp in [1, 2, 4, 8] and mode not in {'slow', 'unsupported'}",
        "tp in (1, 2, 4, 8)",
        "((tp + 2) * 3 - 4) // 2 % 4 == 3",
        "tp / 2 == 2",
        "ratio >= 0.5 and cmd_args.optional == None",
        "+tp == 4 and -offset == -2",
        "values[0] == tp and cmd_args['tp'] == tp",
    ],
)
def test_constraints_support_documented_expression_syntax(expression: str) -> None:
    context = {
        "cmd_args": {
            "enabled": True,
            "disabled": False,
            "mode": "fast",
            "tp": 4,
            "offset": 2,
            "optional": None,
            "ratio": 0.5,
            "values": [4],
        },
        "extra_env_vars": {},
        "system": {"gpus_per_node": 8},
        "test_run": {"num_nodes": 2},
    }

    # Direct names outside the four roots must be declared as aliases.
    constraints = DSEConstraints(
        variables={
            name: f"cmd_args.{name}" for name in ("enabled", "disabled", "mode", "tp", "offset", "ratio", "values")
        },
        expressions={"supported": expression},
    )

    assert constraints.evaluate(context) == (True, None, None)


@pytest.mark.parametrize("expression", ["1 / 0 > 0", "cmd_args.tp + 'x' > 0", "cmd_args.values[2] == 1"])
def test_constraints_wrap_runtime_evaluation_errors(expression: str) -> None:
    constraints = DSEConstraints(expressions={"broken": expression})
    context = {
        "cmd_args": {"tp": 4, "values": [1]},
        "extra_env_vars": {},
        "system": {},
        "test_run": {},
    }

    with pytest.raises(ConstraintEvaluationError, match="Failed to evaluate DSE constraint 'broken'"):
        constraints.evaluate(context)


def test_constraints_require_boolean_result() -> None:
    constraints = DSEConstraints(expressions={"not_a_predicate": "cmd_args.tp + 1"})

    with pytest.raises(ConstraintEvaluationError, match="did not evaluate to a Boolean"):
        constraints.evaluate({"cmd_args": {"tp": 4}, "extra_env_vars": {}, "system": {}, "test_run": {}})


def test_constraints_report_unresolvable_variable_path() -> None:
    constraints = DSEConstraints(
        variables={"tp": "cmd_args.tensor_parallel_size"},
        expressions={"missing": "tp <= 8"},
    )

    with pytest.raises(ConstraintEvaluationError, match=r"cmd_args\.tensor_parallel_size"):
        constraints.evaluate({"cmd_args": {}, "extra_env_vars": {}, "system": {}, "test_run": {}})


def test_test_definition_stops_at_first_declarative_failure(caplog: pytest.LogCaptureFixture) -> None:
    test = SleepTestDefinition(
        name="sleep",
        description="test",
        test_template_name="Sleep",
        cmd_args=SleepCmdArgs(seconds=5),
        dse_constraints=DSEConstraints(
            variables={"seconds": "cmd_args.seconds"},
            expressions={
                "duration_limit": "seconds <= 4",
                "not_evaluated": "1 / 0 > 1",
            },
        ),
    )
    test_run = TestRun(name="sleep", test=test, num_nodes=1, nodes=[])

    with caplog.at_level("INFO"), patch.object(SleepTestDefinition, "constraint_check") as workload_check:
        assert not test.check_constraints(test_run, None)
    workload_check.assert_not_called()
    assert "DSE constraint 'duration_limit' rejected" in caplog.text


def test_test_definition_delegates_to_workload_check_after_declarative_success() -> None:
    test = SleepTestDefinition(
        name="sleep",
        description="test",
        test_template_name="Sleep",
        cmd_args=SleepCmdArgs(seconds=5),
        dse_constraints=DSEConstraints(expressions={"positive_duration": "cmd_args.seconds > 0"}),
    )
    test_run = TestRun(name="sleep", test=test, num_nodes=1, nodes=[])

    with patch.object(SleepTestDefinition, "constraint_check", return_value=False) as workload_check:
        assert not test.check_constraints(test_run, None)
    workload_check.assert_called_once_with(test_run, None)


def test_test_definition_accepts_shared_aliases_and_multiple_expressions_from_toml() -> None:
    test = SleepTestDefinition.model_validate(
        toml.loads(
            """
            name = "sleep"
            description = "test"
            test_template_name = "Sleep"

            [cmd_args]
            seconds = 5

            [dse_constraints.variables]
            seconds = "cmd_args.seconds"
            nodes = "test_run.num_nodes"

            [dse_constraints.expressions]
            positive_duration = "seconds > 0"
            duration_fits = "seconds <= nodes * 5"
            """
        )
    )
    test_run = TestRun(name="sleep", test=test, num_nodes=1, nodes=[])

    assert test.dse_constraints is not None
    assert list(test.dse_constraints.expressions) == ["positive_duration", "duration_fits"]
    assert test.check_constraints(test_run, None)
