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

import logging
from typing import List, Optional, Union, cast

from pydantic import Field, ValidationInfo, field_validator

from cloudai.core import DockerImage, GitRepo, Installable, PythonExecutable
from cloudai.models.workload import CmdArgs, TestDefinition


class MegatronBridgeCmdArgs(CmdArgs):
    """Megatron-Bridge launcher arguments (translated into `setup_experiment.py` flags)."""

    # Slurm/launcher-level
    gpu_type: str = Field(default="gb200")
    log_dir: str = Field(default="")
    time_limit: str = Field(default="00:30:00")
    container_image: str = Field(default="")
    num_gpus: int = Field(default=8)
    gpus_per_node: int = Field(default=8)
    custom_mounts: Optional[str] = Field(default=None)
    enable_vboost: Optional[bool] = Field(default=False)
    dryrun: Optional[bool] = Field(default=False)
    enable_nsys: Optional[bool] = Field(default=False)
    detach: Optional[bool] = Field(default=None)

    # Model/task
    model_name: str = Field(min_length=1)
    model_size: str = Field(min_length=1)
    domain: str = Field(default="llm")
    task: str = Field(default="pretrain")
    compute_dtype: str = Field(default="bf16")
    fp8_recipe: Optional[str] = Field(default=None)
    hf_token: Optional[str] = Field(default=None)
    nemo_home: Optional[str] = Field(default=None)
    wandb_key: Optional[str] = Field(default=None)
    wandb_prj_name: Optional[str] = Field(default=None)
    wandb_exp_name: Optional[str] = Field(default=None)

    # Feature flags (allow sweeps)
    use_tokendrop: Optional[Union[bool, List[bool]]] = Field(default=None)
    use_megatron_fsdp: Optional[Union[bool, List[bool]]] = Field(default=None)
    cuda_graph_impl: Optional[str] = Field(default=None)
    cuda_graph_scope: Optional[Union[str, List[str]]] = Field(default=None)

    # Parallelism
    tp: Optional[Union[int, List[int]]] = Field(default=None)
    pp: Optional[Union[int, List[int]]] = Field(default=None)
    cp: Optional[Union[int, List[int]]] = Field(default=None)
    vp: Optional[Union[int, List[int]]] = Field(default=None)
    ep: Optional[Union[int, List[int]]] = Field(default=None)
    et: Optional[Union[int, List[int]]] = Field(default=None)

    # Batch sizes
    mb: Optional[Union[int, List[int]]] = Field(default=None)
    gb: Optional[Union[int, List[int]]] = Field(default=None)

    # Perf/tuning
    moe_a2a_overlap: Optional[Union[bool, List[bool]]] = Field(default=None)
    max_steps: Optional[int] = Field(default=50)
    recompute_num_layers: Optional[Union[int, List[int]]] = Field(default=None)
    activation_offload_layers: Optional[Union[int, List[int]]] = Field(default=None)
    recompute_modules: Optional[Union[str, List[str]]] = Field(default=None)

    # Optional distributed optimizer instances (for constraints/divisor)
    num_distributed_optimizer_instances: Optional[int] = Field(default=None)

    @field_validator("hf_token", mode="after")
    @classmethod
    def validate_hf_token(cls, v: Optional[str]) -> Optional[str]:
        token = (v or "").strip()
        if not token:
            raise ValueError("cmd_args.hf_token is required. Please set it to your literal HF token string.")
        return token

    @field_validator("model_name", "model_size", mode="after")
    @classmethod
    def validate_model_fields(cls, v: str, info: ValidationInfo) -> str:
        s = v.strip()
        if not s:
            raise ValueError(f"cmd_args.{info.field_name} cannot be empty.")
        return s


class MegatronBridgeTestDefinition(TestDefinition):
    """Megatron-Bridge test definition (CloudAI-managed install + Slurm submission via launcher)."""

    cmd_args: MegatronBridgeCmdArgs

    nemo_run_repo: GitRepo = GitRepo(
        url="https://github.com/NVIDIA-NeMo/Run.git",
        commit="main",
    )

    _docker_image: Optional[DockerImage] = None
    _python_executable: Optional[PythonExecutable] = None
    _megatron_bridge_repo: Optional[GitRepo] = None

    @staticmethod
    def _select_megatron_bridge_repo(git_repos: list[GitRepo]) -> GitRepo | None:
        """Return the Megatron-Bridge repo from `git_repos` (normalized to mount_as=/opt/Megatron-Bridge)."""
        for repo in git_repos:
            if "Megatron-Bridge" in repo.url or (repo.mount_as or "").rstrip("/") == "/opt/Megatron-Bridge":
                return repo if repo.mount_as else repo.model_copy(update={"mount_as": "/opt/Megatron-Bridge"})
        return None

    @field_validator("git_repos", mode="after")
    @classmethod
    def validate_git_repos_has_megatron_bridge_repo(cls, v: list[GitRepo]) -> list[GitRepo]:
        """MegatronBridge requires users to pin the Megatron-Bridge repo version via `[[git_repos]]`."""
        if not v:
            raise ValueError(
                "MegatronBridge requires the user to pin the Megatron-Bridge repository via `[[git_repos]]` "
                "in the test TOML (provide at least url and commit)."
            )

        if cls._select_megatron_bridge_repo(v) is None:
            raise ValueError(
                "MegatronBridge requires `[[git_repos]]` to include the Megatron-Bridge repo (url containing "
                "'Megatron-Bridge' or mount_as='/opt/Megatron-Bridge')."
            )
        return v

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.container_image or "")
        return self._docker_image

    @property
    def python_executable(self) -> PythonExecutable:
        if not self._python_executable:
            self._python_executable = PythonExecutable(git_repo=self.nemo_run_repo)
        return self._python_executable

    @property
    def megatron_bridge_repo(self) -> GitRepo:
        if self._megatron_bridge_repo is None:
            selected = self._select_megatron_bridge_repo(self.git_repos)
            if selected is None:
                raise ValueError(
                    "MegatronBridge requires the user to pin the Megatron-Bridge repository via `[[git_repos]]` "
                    "in the test TOML (provide at least url and commit)."
                )
            self._megatron_bridge_repo = selected
        return self._megatron_bridge_repo

    @property
    def installables(self) -> list[Installable]:
        items: list[Installable] = [self.python_executable, self.megatron_bridge_repo]
        if self.cmd_args.container_image:
            items.insert(0, self.docker_image)
        return items

    def constraint_check(self, tr) -> bool:  # type: ignore[override]  # noqa: C901
        num_gpus = cast(int, self.cmd_args.num_gpus)

        def _as_int(val: Optional[Union[int, List[int]]]) -> Optional[int]:
            return cast(Optional[int], val)

        def _as_bool(val: Optional[Union[bool, List[bool]]]) -> bool:
            return bool(val) if val is not None else False

        def _normalize_str_list(val: Optional[Union[str, List[str]]]) -> list[str]:
            if val is None:
                return []
            if isinstance(val, list):
                items: list[str] = []
                for raw in val:
                    s = str(raw).strip().strip("\"'")
                    if s.startswith("[") and s.endswith("]"):
                        s = s[1:-1]
                    for seg in s.split(","):
                        seg = seg.strip().strip("\"'").lower()
                        if seg:
                            items.append(seg)
                return items
            s = str(val).strip().strip("\"'")
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            items = [seg.strip().strip("\"'").lower() for seg in s.split(",") if seg.strip()]
            return items

        tp = _as_int(self.cmd_args.tp) or 1
        pp = _as_int(self.cmd_args.pp) or 1
        cp = _as_int(self.cmd_args.cp) or 1
        vp = _as_int(self.cmd_args.vp)
        mbs = _as_int(self.cmd_args.mb)
        gbs = _as_int(self.cmd_args.gb)
        etp = _as_int(self.cmd_args.et)
        epv = _as_int(self.cmd_args.ep)

        denom = tp * pp * cp
        dp = num_gpus // denom if denom > 0 else 0

        # Constraint 1: TP/PP/CP divisibility for DP computation
        if denom == 0:
            constraint1 = False
            logging.error("Constraint 1 failed: tp*pp*cp must be > 0. tp=%s pp=%s cp=%s", tp, pp, cp)
        else:
            constraint1 = (num_gpus % denom) == 0
            if not constraint1:
                logging.error(
                    "Constraint 1 failed: num_gpus %% (tp*pp*cp) != 0. num_gpus=%s tp=%s pp=%s cp=%s",
                    num_gpus,
                    tp,
                    pp,
                    cp,
                )

        # Constraint 2: VP validity vs layers (relaxed: num_layers not available in this workload API)
        constraint2 = True  # simple variant doesn't carry num_layers; assume satisfied

        # Constraint 3: DP must be non-zero
        constraint3 = dp != 0
        if not constraint3:
            logging.error("Constraint 3 failed: dp == 0. dp=%s num_gpus=%s tp=%s pp=%s cp=%s", dp, num_gpus, tp, pp, cp)

        # Constraint 4: GBS must be divisible by MBS*DP (and *NDO when FSDP)
        fsdp = _as_bool(self.cmd_args.use_megatron_fsdp)
        ndo = _as_int(self.cmd_args.num_distributed_optimizer_instances) or 1
        extra = ndo if fsdp else 1
        if mbs and gbs:
            divisor = (mbs * dp * extra) if dp != 0 else 0
            constraint4 = (gbs % divisor == 0) if divisor != 0 else False
            if not constraint4:
                logging.error(
                    "Constraint 4 failed: gbs %% (mbs * dp%s) != 0. gbs=%s mbs=%s dp=%s ndo=%s fsdp=%s",
                    " * ndo" if fsdp else "",
                    gbs,
                    mbs,
                    dp,
                    ndo,
                    fsdp,
                )
        else:
            constraint4 = True

        # Determine CUDA graphs enabled from impl/scope (normalized)
        cgi = (self.cmd_args.cuda_graph_impl or "").strip().lower()
        scopes = _normalize_str_list(self.cmd_args.cuda_graph_scope)
        cuda_graphs = (cgi not in {"", "none", "null"}) and len(scopes) > 0

        # Constraint 5: FSDP requires CUDA graphs disabled
        constraint5 = not (fsdp and cuda_graphs)
        if not constraint5:
            logging.error(
                "Constraint 5 failed: use_megatron_fsdp=true requires cuda_graphs=false. fsdp=%s cuda_graphs=%s",
                fsdp,
                cuda_graphs,
            )

        # Constraint 6: FSDP requires PP=1, CP=1, VP=1 (when VP set)
        constraint6 = pp == 1 and cp == 1 and (vp == 1 if vp is not None else True) if fsdp else True
        if not constraint6:
            logging.error(
                "Constraint 6 failed: with fsdp=true, require pp==1, cp==1, vp==1. pp=%s cp=%s vp=%s", pp, cp, vp
            )

        # Constraint 7: MoE parallelism divisibility
        if etp is None or epv is None:
            constraint7 = True
        else:
            moe_denom = etp * epv * pp
            if moe_denom == 0:
                constraint7 = False
                logging.error("Constraint 7 failed: et*ep*pp must be > 0. et=%s ep=%s pp=%s", etp, epv, pp)
            else:
                constraint7 = (num_gpus % moe_denom) == 0
                if not constraint7:
                    logging.error(
                        "Constraint 7 failed: num_gpus %% (et * ep * pp) != 0. num_gpus=%s et=%s ep=%s pp=%s",
                        num_gpus,
                        etp,
                        epv,
                        pp,
                    )

        # Constraint 8: VP allowed set (relaxed: num_layers not available)
        constraint8 = True

        # Constraint 9: NDO > 1 only valid for FSDP
        if (
            self.cmd_args.num_distributed_optimizer_instances is not None
            and self.cmd_args.num_distributed_optimizer_instances > 1
            and not fsdp
        ):
            constraint9 = False
            logging.error(
                "Constraint 9 failed: num_distributed_optimizer_instances > 1 requires use_megatron_fsdp=true. "
                "ndo=%s fsdp=%s",
                self.cmd_args.num_distributed_optimizer_instances,
                fsdp,
            )
        else:
            constraint9 = True

        # Constraint 10: PP>1 cannot be combined with CPU offloading
        cpu_off_layers = _as_int(self.cmd_args.activation_offload_layers) or 0
        if pp > 1 and cpu_off_layers > 0:
            constraint10 = False
            logging.error(
                "Constraint 10 failed: pp > 1 cannot be combined with CPU offloading. "
                "pp=%s activation_offload_layers=%s",
                pp,
                cpu_off_layers,
            )
        else:
            constraint10 = True

        # Constraint 11: CUDA graphs require a2a overlap disabled
        a2a_overlap = _as_bool(self.cmd_args.moe_a2a_overlap)
        constraint11 = not (cuda_graphs and a2a_overlap)
        if not constraint11:
            logging.error(
                "Constraint 11 failed: cuda_graphs=true requires moe_a2a_overlap=false. "
                "cuda_graphs=%s moe_a2a_overlap=%s",
                cuda_graphs,
                a2a_overlap,
            )

        # Constraint 12: CUDA graphs not supported with CPU offloading
        if cuda_graphs and cpu_off_layers > 0:
            constraint12 = False
            logging.error(
                "Constraint 12 failed: CUDA graphs not supported with CPU offloading. activation_offload_layers=%s",
                cpu_off_layers,
            )
        else:
            constraint12 = True

        # Constraint 13: cuda_graph_impl validation
        if cgi not in {"", "none", "null", "transformer_engine", "local"}:
            constraint13 = False
            logging.error(
                "Constraint 13 failed: Invalid cuda_graph_impl=%s. Expected one of none, transformer_engine, local.",
                cgi,
            )
        else:
            constraint13 = True

        # Constraint 14: impl/scope compatibility
        allowed_scopes = {"attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"}
        if cgi == "local":
            if len(scopes) > 0 and not (len(scopes) == 1 and scopes[0] == "full_iteration"):
                constraint14 = False
                logging.error(
                    "Constraint 14 failed: cuda_graph_impl=local only allows cuda_graph_scope=['full_iteration']."
                )
            else:
                constraint14 = True
        elif cgi == "transformer_engine":
            has_full = any(s == "full_iteration" for s in scopes)
            invalid = [s for s in scopes if s not in allowed_scopes and s != "full_iteration"]
            if has_full or invalid:
                constraint14 = False
                if has_full:
                    logging.error(
                        "Constraint 14 failed: 'full_iteration' not allowed for transformer_engine cuda graphs."
                    )
                if invalid:
                    logging.error(
                        "Constraint 14 failed: invalid cuda_graph_scope values for transformer_engine: %s", invalid
                    )
            else:
                constraint14 = True
        else:
            constraint14 = True

        # Constraint 15: scope cannot contain both moe and moe_router
        if "moe" in scopes and "moe_router" in scopes:
            constraint15 = False
            logging.error("Constraint 15 failed: cuda_graph_scope must not contain both 'moe' and 'moe_router'.")
        else:
            constraint15 = True

        # Constraint 16: moe_preprocess requires moe_router
        if "moe_preprocess" in scopes and "moe_router" not in scopes:
            constraint16 = False
            logging.error("Constraint 16 failed: 'moe_preprocess' scope requires 'moe_router' scope.")
        else:
            constraint16 = True

        # Constraint 17: recompute vs CUDA graphs incompatibilities
        rec_modules = set(_normalize_str_list(self.cmd_args.recompute_modules))
        if scopes and rec_modules:
            bad = False
            if "attn" in scopes and ({"core_attn", "mla_up_proj"} & rec_modules):
                bad = True
                logging.error(
                    "Constraint 17 failed: attn cuda graph is not supported with core_attn or mla_up_proj recompute. "
                    "recompute_modules=%s",
                    sorted(rec_modules),
                )
            if "mlp" in scopes and "mlp" in rec_modules:
                bad = True
                logging.error("Constraint 17 failed: mlp cuda graph is not supported with mlp recompute.")
            if "moe" in scopes and ({"moe_act", "moe", "shared_experts"} & rec_modules):
                bad = True
                logging.error(
                    "Constraint 17 failed: moe cuda graph is not supported with moe_act/moe/shared_experts recompute."
                )
            if "moe_router" in scopes and ({"moe", "shared_experts"} & rec_modules):
                bad = True
                logging.error(
                    "Constraint 17 failed: moe_router cuda graph is not supported with moe/shared_experts recompute."
                )
            if "layernorm" in rec_modules and (
                "attn" in scopes and "mlp" in scopes and ("moe" in scopes or "moe_router" in scopes)
            ):
                bad = True
                logging.error(
                    "Constraint 17 failed: cuda graph is not supported with layernorm recompute across "
                    "attn+mlp+(moe|moe_router)."
                )
            constraint17 = not bad
        else:
            constraint17 = True

        return bool(
            constraint1
            and constraint2
            and constraint3
            and constraint4
            and constraint5
            and constraint6
            and constraint7
            and constraint8
            and constraint9
            and constraint10
            and constraint11
            and constraint12
            and constraint13
            and constraint14
            and constraint15
            and constraint16
            and constraint17
        )
