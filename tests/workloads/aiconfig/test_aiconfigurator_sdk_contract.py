# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Compatibility canary between ``runtime/predictor.py`` and the pinned aiconfigurator SDK.

``runtime/predictor.py`` is our adapter onto ``aiconfigurator.sdk.*``; the version
is pinned by ``AiconfiguratorTestDefinition.python_environment`` (``aiconfigurator~=0.5.0``).
If a pinned-version bump renames/removes a SDK symbol the adapter relies on, the
adapter breaks at run time and DSE silently stops producing metrics -- with nothing
to warn us, because the rest of the aiconfig tests mock or never touch the SDK.

This canary closes that gap *without* the heavy (~559 MB, torch) install: it fetches
only the pinned source with ``uv pip install --no-deps`` (seconds, no torch) and
audits it with the ``ast`` module -- no imports, so the torch-importing SDK modules
parse fine. It asserts the exact surface ``predictor.py`` consumes still exists.

Scope: this catches input-side drift (renamed/removed fields, enum members, funcs).
It does NOT catch a changed *return* shape (e.g. a renamed summary-df column); that
would need a real prediction run and is out of scope for a lightweight canary.

Marked ``ci_only``: it requires network/``uv`` to fetch the pinned source, so it runs
in CI's dedicated ``-m ci_only`` step rather than the hermetic unit suite. Keep the
contract constants below in lockstep with ``runtime/predictor.py``.
"""

from __future__ import annotations

import ast
import functools
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from cloudai.workloads.aiconfig import AiconfiguratorCmdArgs, AiconfiguratorTestDefinition
from cloudai.workloads.aiconfig.aiconfigurator import Agg

pytestmark = pytest.mark.ci_only

# --- The SDK surface runtime/predictor.py depends on (keep in lockstep with it) ---

# aic_config.ModelConfig(...) / aic_config.RuntimeConfig(...) keyword arguments.
_MODEL_CONFIG_FIELDS = [
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "moe_tp_size",
    "moe_ep_size",
    "gemm_quant_mode",
    "moe_quant_mode",
    "kvcache_quant_mode",
    "fmha_quant_mode",
    "comm_quant_mode",
    "nextn",
    "nextn_accept_rates",
    "overwrite_num_layers",
]
_RUNTIME_CONFIG_FIELDS = ["batch_size", "isl", "osl"]

# common.<Enum>[<member>] resolved by predictor._to_enum from the CLI defaults.
_QUANT_ENUM_MEMBERS = {
    "GEMMQuantMode": "fp8_block",
    "MoEQuantMode": "fp8",
    "KVCacheQuantMode": "fp8",
    "FMHAQuantMode": "fp8",
    "CommQuantMode": "half",
}

# Free functions predictor.py calls, keyed by the sdk-relative source file.
_FREE_FUNCTIONS = {
    "perf_database.py": "get_database",
    "models.py": "get_model",
    "backends/factory.py": "get_backend",
}
_INFERENCE_SESSION_METHODS = ["run_static"]


def _pinned_requirement() -> str:
    """The aiconfigurator requirement string the workload pins (single source of truth)."""
    tdef = AiconfiguratorTestDefinition(
        name="aiconfig",
        description="contract",
        test_template_name="Aiconfigurator",
        cmd_args=AiconfiguratorCmdArgs(
            model_name="LLAMA3.1_70B", system="h200_sxm", isl=1, osl=1, agg=Agg(batch_size=1, ctx_tokens=1)
        ),
    )
    reqs = tdef.python_environment.requirements
    assert len(reqs) == 1, f"expected a single aiconfigurator requirement, got {reqs}"
    return reqs[0]


@functools.lru_cache(maxsize=1)
def _fetch_pinned_sdk_src() -> Path:
    """``uv pip install --no-deps`` the pinned aiconfigurator into a temp dir; return its ``sdk`` path.

    ``--no-deps`` skips the entire torch/cuda tree, so this is seconds and tiny. We
    never import the package (its compute modules need torch); callers AST-parse it.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not available to fetch the pinned aiconfigurator source")

    target = Path(tempfile.mkdtemp(prefix="aiconfigurator_sdk_"))
    proc = subprocess.run(
        [uv, "pip", "install", "--no-deps", "--target", str(target), _pinned_requirement()],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to fetch aiconfigurator --no-deps: {proc.stderr.strip()[:400]}")

    sdk = target / "aiconfigurator" / "sdk"
    if not sdk.is_dir():
        raise RuntimeError(f"aiconfigurator.sdk not found under fetched package at {sdk}")
    return sdk


def _sdk_src() -> Path:
    try:
        return _fetch_pinned_sdk_src()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def _parse(sdk: Path, rel: str) -> ast.Module:
    return ast.parse((sdk / rel).read_text(encoding="utf-8"))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name), None)


def _defines_function(tree: ast.Module, name: str) -> bool:
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree))


def _accepted_kwargs(cls: ast.ClassDef) -> tuple[set[str], bool]:
    """Constructor-accepted kwargs: explicit ``__init__`` params, else annotated (dataclass/pydantic) fields."""
    init = next(
        (b for b in cls.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)) and b.name == "__init__"),
        None,
    )
    if init is not None:
        args = init.args
        names = {p.arg for p in (args.posonlyargs + args.args + args.kwonlyargs) if p.arg != "self"}
        return names, args.kwarg is not None
    fields = {b.target.id for b in cls.body if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name)}
    return fields, False


def _enum_members(cls: ast.ClassDef) -> set[str]:
    members: set[str] = set()
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            members.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            members.add(stmt.target.id)
    return members


@pytest.mark.parametrize(
    "cls_name, expected, rel",
    [
        ("ModelConfig", _MODEL_CONFIG_FIELDS, "config.py"),
        ("RuntimeConfig", _RUNTIME_CONFIG_FIELDS, "config.py"),
    ],
)
def test_config_classes_accept_required_kwargs(cls_name: str, expected: list[str], rel: str) -> None:
    cls = _find_class(_parse(_sdk_src(), rel), cls_name)
    assert cls is not None, f"aiconfigurator.sdk.config.{cls_name} no longer exists"

    accepted, has_var_kwargs = _accepted_kwargs(cls)
    if has_var_kwargs:
        return
    missing = sorted(set(expected) - accepted)
    assert not missing, f"{cls_name} no longer accepts {missing}; predictor.py is incompatible with the pinned SDK"


def test_quant_enums_expose_used_members() -> None:
    tree = _parse(_sdk_src(), "common.py")
    for enum_name, member in _QUANT_ENUM_MEMBERS.items():
        cls = _find_class(tree, enum_name)
        assert cls is not None, f"aiconfigurator.sdk.common.{enum_name} no longer exists"
        members = _enum_members(cls)
        assert member in members, f"{enum_name} no longer defines member {member!r} that predictor.py relies on"


@pytest.mark.parametrize("rel, func", sorted(_FREE_FUNCTIONS.items()))
def test_sdk_free_functions_exist(rel: str, func: str) -> None:
    assert _defines_function(_parse(_sdk_src(), rel), func), (
        f"aiconfigurator.sdk.{rel.replace('/', '.').removesuffix('.py')}.{func} no longer exists"
    )


def test_inference_session_exposes_used_methods() -> None:
    cls = _find_class(_parse(_sdk_src(), "inference_session.py"), "InferenceSession")
    assert cls is not None, "aiconfigurator.sdk.inference_session.InferenceSession no longer exists"

    methods = {b.name for b in cls.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = sorted(set(_INFERENCE_SESSION_METHODS) - methods)
    assert not missing, f"InferenceSession no longer defines {missing} that predictor.py calls"
