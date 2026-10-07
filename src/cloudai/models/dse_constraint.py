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

"""Safe declarative constraints for DSE configurations."""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


class ConstraintEvaluationError(ValueError):
    """Raised when a declarative DSE constraint cannot be evaluated."""


_ROOT_NAMES = {"cmd_args", "extra_env_vars", "system", "test_run"}
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_COMPARISON_OPERATORS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}
_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.BinOp,
    *_BINARY_OPERATORS,
    ast.UnaryOp,
    ast.Not,
    ast.UAdd,
    ast.USub,
    ast.Compare,
    *_COMPARISON_OPERATORS,
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Load,
)


def _parse_expression(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid constraint expression: {e.msg}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise ValueError(f"Constraint expression does not allow {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Constraint expression cannot access private attributes")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (bool, int, float, str, type(None))):
            raise ValueError(f"Constraint expression does not allow {type(node.value).__name__} literals")
    return tree


def _resolve_path(context: Mapping[str, Any], path: str) -> Any:
    value: Any = context
    for component in path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ConstraintEvaluationError(f"Cannot resolve constraint variable path '{path}'")
        value = value[component]
    return value


def _resolve_member(value: Any, key: Any) -> Any:
    if isinstance(value, Mapping):
        try:
            return value[key]
        except (KeyError, TypeError) as e:
            raise ConstraintEvaluationError(f"Cannot resolve constraint member {key!r}") from e
    if isinstance(value, (list, tuple)) and isinstance(key, int) and not isinstance(key, bool):
        try:
            return value[key]
        except IndexError as e:
            raise ConstraintEvaluationError(f"Cannot resolve constraint index {key}") from e
    raise ConstraintEvaluationError(f"Cannot resolve constraint member {key!r}")


def _evaluate_node(node: ast.AST, context: Mapping[str, Any]) -> Any:  # noqa: C901
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, context)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ConstraintEvaluationError(f"Unknown constraint variable '{node.id}'")
        return context[node.id]
    if isinstance(node, ast.Attribute):
        return _resolve_member(_evaluate_node(node.value, context), node.attr)
    if isinstance(node, ast.Subscript):
        return _resolve_member(_evaluate_node(node.value, context), _evaluate_node(node.slice, context))
    if isinstance(node, ast.List):
        return [_evaluate_node(element, context) for element in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_node(element, context) for element in node.elts)
    if isinstance(node, ast.Set):
        return {_evaluate_node(element, context) for element in node.elts}
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_evaluate_node(value, context) for value in node.values)
        return any(_evaluate_node(value, context) for value in node.values)
    if isinstance(node, ast.UnaryOp):
        value = _evaluate_node(node.operand, context)
        if isinstance(node.op, ast.Not):
            return not value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConstraintEvaluationError("Unary arithmetic requires a numeric operand")
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _evaluate_node(node.left, context)
        right = _evaluate_node(node.right, context)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in (left, right)):
            raise ConstraintEvaluationError("Constraint arithmetic requires numeric operands")
        return _BINARY_OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.Compare):
        left = _evaluate_node(node.left, context)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _evaluate_node(comparator, context)
            if not _COMPARISON_OPERATORS[type(op)](left, right):
                return False
            left = right
        return True
    raise ConstraintEvaluationError(f"Unsupported constraint expression element: {type(node).__name__}")


class DSEConstraints(BaseModel):
    """Shared variable bindings and named Boolean constraints for DSE candidates."""

    model_config = ConfigDict(extra="forbid")

    variables: dict[str, str] = Field(default_factory=dict)
    expressions: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        for alias, path in self.variables.items():
            if not alias.isidentifier() or alias.startswith("_"):
                raise ValueError(f"Invalid constraint variable name: {alias!r}")
            if alias in _ROOT_NAMES:
                raise ValueError(f"Constraint variable name is reserved: {alias!r}")
            components = path.split(".")
            if (
                len(components) < 2
                or components[0] not in _ROOT_NAMES
                or any(not component or component.startswith("_") for component in components)
            ):
                raise ValueError(
                    "Constraint variable path must start with one of "
                    f"{sorted(_ROOT_NAMES)} and contain only public, nonempty components: {path!r}"
                )

        for name, expression in self.expressions.items():
            if not name.strip():
                raise ValueError("Constraint names cannot be empty")
            tree = _parse_expression(expression)
            referenced_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            unknown_names = referenced_names - _ROOT_NAMES - self.variables.keys()
            if unknown_names:
                raise ValueError(f"Constraint '{name}' uses unknown variables: {', '.join(sorted(unknown_names))}")
        return self

    def evaluate(self, context: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
        """Evaluate constraints in declaration order and return the first failure."""
        evaluation_context = dict(context)
        try:
            evaluation_context.update({alias: _resolve_path(context, path) for alias, path in self.variables.items()})
        except ConstraintEvaluationError as e:
            raise ConstraintEvaluationError(f"Failed to resolve DSE constraint variables: {e}") from e

        for name, expression in self.expressions.items():
            try:
                result = _evaluate_node(_parse_expression(expression), evaluation_context)
            except (ArithmeticError, ConstraintEvaluationError, TypeError, ValueError) as e:
                raise ConstraintEvaluationError(f"Failed to evaluate DSE constraint '{name}': {e}") from e
            if not isinstance(result, bool):
                raise ConstraintEvaluationError(f"DSE constraint '{name}' did not evaluate to a Boolean")
            if not result:
                return False, name, expression
        return True, None, None
