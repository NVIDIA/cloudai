# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import nemo_run as run
from nemo.collections.llm.recipes.llama3_8b import pretrain_recipe as pretrain_recipe_8b
from nemo.collections.llm.recipes.llama3_70b import pretrain_recipe as pretrain_recipe_70b
from nemo.collections.llm.recipes.llama31_405b import pretrain_recipe as pretrain_recipe_405b
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin
from nemo.utils import logging

from ..utils import args_sanity_check, get_comm_overlap_callback_idx, hf_tokenizer, set_primary_perf_configs
from .llama31_utils import llama31_auto_configs, llama31_parse_cli_args, llama31_slurm_executor


def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    num_layers: int,
    enable_cuda_graphs: bool,
    max_steps,
):
    match args.model_size:
        case "8b":
            recipe = pretrain_recipe_8b(performance_mode=True)
            HF_MODEL_URI = "meta-llama/Meta-Llama-3-8B"
        case "70b":
            recipe = pretrain_recipe_70b(performance_mode=True)
            HF_MODEL_URI = "meta-llama/Meta-Llama-3-70B"
            ub_cfg = {
                "h100": {
                    "bf16": userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
                    "fp8": userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
                },
                "b200": {
                    "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
                    "fp8": userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
                },
            }
        case "405b":
            recipe = pretrain_recipe_405b(performance_mode=True)
            HF_MODEL_URI = "meta-llama/Llama-3.1-405B"
            ub_cfg = {
                "h100": {
                    "bf16": userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
                    "fp8": userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
                },
                "b200": {
                    "bf16": userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
                    "fp8": userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
                },
            }

    recipe = set_primary_perf_configs(
        recipe,
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
    )

    gpu_type = args.gpu.lower()

    # data module configs
    recipe.data.num_train_samples = max_steps * gbs * mbs  # ensure only 1 epoch for whole run
    recipe.data.tokenizer = hf_tokenizer(HF_MODEL_URI)

    # compute dtype configs
    if args.compute_dtype.lower() == "fp8":
        recipe.trainer.plugins = bf16_with_fp8_mixed()
        recipe.trainer.plugins.grad_reduce_in_fp32 = False

    if args.model_size != "8b":
        comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
        assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

        tp_comm_overlap_cfg = ub_cfg[gpu_type][args.compute_dtype]
        # needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
        tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
        recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg

    recipe.model.config.enable_cuda_graph = enable_cuda_graphs
    recipe.trainer.strategy.use_te_rng_tracker = enable_cuda_graphs

    if num_layers:
        recipe.model.config.num_layers = num_layers

    # Disable validation as it causes hangs
    recipe.trainer.limit_val_batches = 0

    # Optimization code
    if args.optimization_name:
        exec(args.optimization_code)

    return recipe


if __name__ == "__main__":
    args = llama31_parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = kwargs = llama31_auto_configs(args)

    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, num_layers, enable_cuda_graphs, max_steps = kwargs

    recipe = override_recipe_configs(
        args,
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        num_layers,
        enable_cuda_graphs,
        max_steps,
    )

    exp_name = "_".join(
        [
            f"{args.cluster}",
            f"{args.gsw_version}",
            f"{args.compute_dtype}",
            "llama3.1",
            f"{args.model_size}",
            f"{args.num_gpus}",
        ]
    )

    executor = llama31_slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars=args.custom_env_vars,
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
    )
    plugins = [PerfEnvPlugin(enable_vboost=True, nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None)]

    if args.enable_profiling:
        if args.model_size == "8b":
            nsys_ranks = [0, 1, 2, 3, 4, 5, 6, 7]
        else:
            nsys_ranks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        NSYS_STEPS = 30
        nsys_recipe = override_recipe_configs(
            args,
            num_nodes,
            mbs,
            gbs,
            tp_size,
            pp_size,
            cp_size,
            vp_size,
            ep_size,
            num_layers,
            enable_cuda_graphs,
            NSYS_STEPS,
        )

        # todo: nvbug: multitank profiling crashes
        nsys_plugins = [*plugins, NsysPlugin(start_step=20, end_step=30)]

    if args.enable_nccltrace:
        # Figure out how to print this inside .out log
        INFO_STR = (
            f"GSW: MODEL=llama3.1 FRAMEWORK=nemo MODEL_SIZE={args.model_size} "
            f"JOB_NUM_NODES={num_nodes} GPUS_PER_NODE={args.gpus_per_node} "
            f"DTYPE=${args.compute_dtype} SYNTHETIC_DATA=true "
            f"GSW_VERSION={args.gsw_version} FW_VERSION={args.framework_version} "
            f"IMAGE={args.container_image} JOB_ID=TODO JOB_MODE=training "
            f"OPTIMIZATION_NAME={args.optimization_name} "
            f"OPTIMIZATION_CODE={args.optimization_code} BASE_CONFIG=TODO"
        )
        logging.info(INFO_STR)

        env_vars = {
            "NCCL_DEBUG_SUBSYS": "COLL,P2P,NET",
            "NCCL_DEBUG": "INFO",
        }
        env_vars |= args.custom_env_vars
        nccltrace_executor = llama31_slurm_executor(
            args.account,
            args.partition,
            args.log_dir,
            num_nodes,
            args.gpus_per_node,
            time_limit="00:15:00",
            container_image=args.container_image,
            custom_mounts=args.custom_mounts,
            custom_env_vars=env_vars,
            hf_token=args.hf_token,
            nemo_home=args.nemo_home,
            wandb_key=args.wandb_key,
        )
        NCCL_STEPS = 5
        nccltrace_recipe = override_recipe_configs(
            args,
            num_nodes,
            mbs,
            gbs,
            tp_size,
            pp_size,
            cp_size,
            vp_size,
            ep_size,
            num_layers,
            enable_cuda_graphs,
            NCCL_STEPS,
        )

    with run.Experiment(exp_name) as exp:
        if not args.disable_perfrun:
            exp.add(
                recipe,
                executor=executor,
                name=exp_name,
                plugins=plugins,
            )
        if args.enable_profiling:
            exp.add(
                nsys_recipe,
                executor=executor,
                name=exp_name + "_nsys",
                plugins=nsys_plugins,
            )
        if args.enable_nccltrace:
            exp.add(
                nccltrace_recipe,
                executor=nccltrace_executor,
                name=exp_name + "_nccltrace",
                plugins=plugins,
            )
        if not args.dryrun:
            exp.run(sequential=False, detach=True)
        else:
            exp.dryrun()
