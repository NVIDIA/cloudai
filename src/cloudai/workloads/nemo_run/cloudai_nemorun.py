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

import os
from typing import Optional

import lightning.pytorch as pl
import nemo_run as run
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.collections.llm.gpt.model.llama import Llama3Config70B, Llama31Config405B, LlamaModel
from nemo.collections.llm.gpt.model.nemotron import Nemotron4Config15B, Nemotron4Config340B, NemotronModel
from nemo.collections.llm.recipes.nemotron3_8b import pretrain_recipe as nemotron3_8b_recipe
from nemo.collections.llm.recipes.qwen3_30b_a3b import pretrain_recipe as qwen3_30b_a3b_pretrain_recipe
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    BulkOverlapCfg,
    PipelineOverlapCfg,
    RingExchangeOverlapCfg,
    TransformerLayerTPOverlapCfg,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import AutoResume, NeMoLogger
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.callbacks.moe_token_drop import MegatronTokenDropCallback
from nemo.lightning.pytorch.callbacks.nsys import NsysCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.utils.exp_manager import TimingCallback


def set_enable_cuda_graphs_params(recipe):
    enable_cuda_graphs = os.getenv("CLOUDAI_ENABLE_CUDA_GRAPHS", "0") == "1"
    if enable_cuda_graphs:
        recipe.model.config.enable_cuda_graph = True
        recipe.trainer.strategy.use_te_rng_tracker = True


@run.cli.factory(is_target_default=True)
def default_log(
    dir: Optional[str] = None,
    name: str = "default",
    tensorboard_logger: Optional[run.Config[TensorBoardLogger]] = None,
    wandb_logger: Optional[run.Config[WandbLogger]] = None,
) -> run.Config[NeMoLogger]:
    # Default TensorBoard logger if not provided
    if tensorboard_logger is None:
        tensorboard_logger = run.Config(TensorBoardLogger, save_dir="tb_logs", name=name)

    return run.Config(
        NeMoLogger,
        ckpt=None,
        name=name,
        tensorboard=tensorboard_logger,
        wandb=wandb_logger,
        log_dir=dir,
    )


@run.cli.factory(is_target_default=True)
def default_resume(resume_if_exists=True, resume_ignore_no_checkpoint=True) -> run.Config[AutoResume]:
    return run.Config(
        AutoResume,
        resume_if_exists=resume_if_exists,
        resume_ignore_no_checkpoint=resume_ignore_no_checkpoint,
    )


@run.cli.factory
@run.autoconvert
def hf_tokenizer_llama3_8b() -> run.Config[AutoTokenizer]:
    model_name = "meta-llama/Meta-Llama-3-8B"
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


@run.cli.factory
@run.autoconvert
def hf_tokenizer_llama3_70b() -> run.Config[AutoTokenizer]:
    model_name = "meta-llama/Meta-Llama-3-70B"
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


@run.cli.factory
@run.autoconvert
def hf_tokenizer_llama3_405b() -> run.Config[AutoTokenizer]:
    model_name = "meta-llama/Llama-3.1-405B"
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


@run.cli.factory
@run.autoconvert
def hf_tokenizer_nemotron3_8b() -> run.Config[AutoTokenizer]:
    model_name = "nvidia/nemotron-3-8b"
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


@run.cli.factory
@run.autoconvert
def hf_tokenizer_nemotron4_15b() -> run.Config[AutoTokenizer]:
    model_name = "nvidia/nemotron-4-15b"
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


@run.cli.factory
@run.autoconvert
def hf_tokenizer_nemotron4_340b() -> run.Config[AutoTokenizer]:
    model_name = "nvidia/nemotron-4-340b"
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )


@run.cli.factory(target=TokenizerSpec)
@run.autoconvert
def null_tokenizer(vocab_size: int = 256000) -> run.Config[TokenizerSpec]:
    return run.Config(get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=vocab_size)


@run.cli.factory
@run.autoconvert
def timing_callback() -> run.Config[TimingCallback]:
    return run.Config(TimingCallback)


@run.cli.factory
@run.autoconvert
def garbage_collection_callbacks() -> list[pl.Callback]:
    return [timing_callback(), run.Config(GarbageCollectionCallback, gc_interval_train=100, gc_interval_val=100)]


@run.cli.factory
@run.autoconvert
def nsys_callbacks() -> list[pl.Callback]:
    start_step = 5
    end_step = 10
    return [
        timing_callback(),
        run.Config(
            NsysCallback,
            start_step=start_step,
            end_step=end_step,
        ),
    ]


@run.cli.factory
@run.autoconvert
def comms_overlap_callbacks() -> list[pl.Callback]:
    return [
        timing_callback(),
        run.Config(MegatronCommOverlapCallback, tp_comm_overlap=False),
    ]


@run.cli.factory
@run.autoconvert
def llama3_70b_bf16_h100_tp_overlap_config() -> run.Config[TransformerLayerTPOverlapCfg]:
    return run.Config(
        TransformerLayerTPOverlapCfg,
        qkv_dgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=4,
            set_sm_margin=False,
        ),
        qkv_wgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=24,
            set_sm_margin=False,
        ),
        qkv_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_fprop=run.Config(
            PipelineOverlapCfg,
            num_sm=24,
            cga_size=2,
            num_splits=4,
            set_sm_margin=True,
            fp8_buf=False,
        ),
        fc1_dgrad=run.Config(
            BulkOverlapCfg,
            num_sm=2,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_wgrad=run.Config(
            BulkOverlapCfg,
            num_sm=4,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_fprop=run.Config(
            PipelineOverlapCfg,
            num_sm=16,
            cga_size=2,
            num_splits=4,
            set_sm_margin=True,
            fp8_buf=False,
        ),
    )


@run.cli.factory
@run.autoconvert
def llama3_70b_bf16_b200_tp_overlap_config() -> run.Config[TransformerLayerTPOverlapCfg]:
    return run.Config(
        TransformerLayerTPOverlapCfg,
        qkv_dgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=8,
            set_sm_margin=False,
        ),
        qkv_wgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=24,
            set_sm_margin=False,
        ),
        qkv_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_fprop=run.Config(
            PipelineOverlapCfg,
            num_sm=32,
            cga_size=2,
            num_splits=4,
            set_sm_margin=True,
            fp8_buf=False,
            method="pipeline",
        ),
        fc1_dgrad=run.Config(
            BulkOverlapCfg,
            num_sm=2,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_wgrad=run.Config(
            BulkOverlapCfg,
            num_sm=2,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_fprop=run.Config(
            PipelineOverlapCfg,
            num_sm=8,
            cga_size=2,
            num_splits=4,
            set_sm_margin=True,
            fp8_buf=False,
            method="pipeline",
        ),
    )


@run.cli.factory
@run.autoconvert
def llama3_70b_fp8_b200_tp_overlap_config() -> run.Config[TransformerLayerTPOverlapCfg]:
    return run.Config(
        TransformerLayerTPOverlapCfg,
        qkv_dgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=4,
            set_sm_margin=False,
        ),
        qkv_wgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=24,
            set_sm_margin=False,
        ),
        qkv_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=True,
        ),
        fc1_dgrad=run.Config(
            BulkOverlapCfg,
            num_sm=2,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_wgrad=run.Config(
            BulkOverlapCfg,
            num_sm=4,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=True,
        ),
    )


@run.cli.factory
@run.autoconvert
def llama3_405b_fp8_b200_tp_overlap_config() -> run.Config[TransformerLayerTPOverlapCfg]:
    return run.Config(
        TransformerLayerTPOverlapCfg,
        qkv_dgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=8,
            set_sm_margin=False,
        ),
        qkv_wgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=32,
            set_sm_margin=False,
        ),
        qkv_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=True,
        ),
        fc1_dgrad=run.Config(
            BulkOverlapCfg,
            num_sm=2,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_wgrad=run.Config(
            BulkOverlapCfg,
            num_sm=8,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=True,
        ),
    )


@run.cli.factory
@run.autoconvert
def llama3_70b_fp8_h100_tp_overlap_config() -> run.Config[TransformerLayerTPOverlapCfg]:
    return run.Config(
        TransformerLayerTPOverlapCfg,
        qkv_dgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=4,
            set_sm_margin=False,
        ),
        qkv_wgrad=run.Config(
            BulkOverlapCfg,
            cga_size=2,
            method="bulk",
            num_sm=4,
            set_sm_margin=False,
        ),
        qkv_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        proj_fprop=run.Config(
            PipelineOverlapCfg,
            num_sm=24,
            cga_size=2,
            num_splits=4,
            set_sm_margin=True,
            fp8_buf=True,
            method="pipeline",
        ),
        fc1_dgrad=run.Config(
            BulkOverlapCfg,
            num_sm=2,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_wgrad=run.Config(
            BulkOverlapCfg,
            num_sm=4,
            cga_size=2,
            set_sm_margin=False,
            method="bulk",
        ),
        fc1_fprop=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_dgrad=run.Config(
            RingExchangeOverlapCfg,
            aggregate=False,
            method="ring_exchange",
            num_sm=1,
            set_sm_margin=False,
        ),
        fc2_fprop=run.Config(
            PipelineOverlapCfg,
            num_sm=16,
            cga_size=2,
            num_splits=4,
            set_sm_margin=True,
            fp8_buf=False,
            method="pipeline",
        ),
    )


def get_tp_overlap_config():
    gpu_type = os.getenv("CLOUDAI_GPU_TYPE")
    compute_dtype = os.getenv("CLOUDAI_GPU_DTYPE")
    if gpu_type == "h100" and compute_dtype == "bf16":
        tp_overlap_cfg = llama3_70b_bf16_h100_tp_overlap_config()
        tp_comm_overlap = True
    elif gpu_type == "h100" and compute_dtype == "fp8":
        tp_overlap_cfg = llama3_70b_fp8_h100_tp_overlap_config()
        tp_comm_overlap = True
    elif gpu_type == "b200" and compute_dtype == "bf16":
        tp_overlap_cfg = llama3_70b_bf16_b200_tp_overlap_config()
        tp_comm_overlap = True
    elif gpu_type in ["b200", "gb200"] and compute_dtype == "fp8":
        tp_overlap_cfg = llama3_405b_fp8_b200_tp_overlap_config()
        tp_comm_overlap = True
    else:
        print(
            "Warning: Not using Default Comm Overlap Config.\n"
            "Please set the GPU type and compute dtype in the environment variables."
        )
        tp_overlap_cfg = None
        tp_comm_overlap = False
    return tp_overlap_cfg, tp_comm_overlap


def set_perf_optimization_configs(recipe):
    recipe.model.config.cross_entropy_fusion_impl = "te"

    is_ddp_obj = hasattr(recipe.trainer.strategy, "ddp") and not isinstance(recipe.trainer.strategy.ddp, str)
    if is_ddp_obj:
        # Disable local gradient checker at non-debugging mode
        recipe.trainer.strategy.ddp.check_for_nan_in_grad = False
        recipe.trainer.strategy.ddp.check_for_large_grads = False

    return recipe


# LLAMA3 8B Recipe
@run.cli.factory(target=llm.pretrain)
def cloudai_llama3_8b_recipe() -> run.Partial:
    from nemo.collections.llm.recipes.llama3_8b import pretrain_recipe

    recipe = pretrain_recipe(performance_mode=True)

    recipe.data.tokenizer = null_tokenizer(vocab_size=128256)
    recipe.log = default_log()
    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="llama3",
        )
    )
    set_enable_cuda_graphs_params(recipe)
    recipe.trainer.strategy.cross_entropy_fusion_impl = "te"
    return recipe


# LLAMA3 70B Recipe
@run.cli.factory(target=llm.pretrain)
def cloudai_llama3_70b_recipe() -> run.Partial:
    recipe = run.Partial(
        llm.pretrain,
        model=run.Config(LlamaModel, config=Llama3Config70B()),
        data=run.Config(
            MockDataModule,
            seq_length=8192,
            micro_batch_size=1,
            global_batch_size=8,
            tokenizer=hf_tokenizer_llama3_70b(),
        ),
        trainer=run.Config(
            nl.Trainer,
            devices=8,
            num_nodes=1,
            accelerator="gpu",
            max_steps=10,
            limit_test_batches=50,
            limit_val_batches=32,
            log_every_n_steps=10,
            accumulate_grad_batches=1,
            plugins=run.Config(
                nl.MegatronMixedPrecision,
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                autocast_enabled=False,
                grad_reduce_in_fp32=False,
            ),
            strategy=run.Config(
                nl.MegatronStrategy,
                ckpt_async_save=True,
                ckpt_parallel_load=True,
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                sequence_parallel=True,
                pipeline_dtype=torch.bfloat16,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                ),
                gradient_as_bucket_view=True,
            ),
            num_sanity_val_steps=0,
            val_check_interval=1000,
            max_epochs=10,
            callbacks=[
                timing_callback(),
            ],
        ),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=3e-4,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                weight_decay=0.1,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-05,
                clip_grad=1.0,
                fp16=False,
            ),
            lr_scheduler=run.Config(
                CosineAnnealingScheduler,
                warmup_steps=2000,
                constant_steps=0,
                min_lr=2.9999999999999997e-05,
            ),
        ),
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
    )
    recipe.model.config.vocab_size = 128256
    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="llama3",
        )
    )
    recipe.trainer.strategy.cross_entropy_fusion_impl = "te"
    set_enable_cuda_graphs_params(recipe)

    tp_overlap_cfg, tp_comm_overlap = get_tp_overlap_config()

    recipe.trainer.callbacks.append(
        run.Config(
            MegatronCommOverlapCallback,
            tp_comm_overlap=tp_comm_overlap,
            tp_comm_overlap_cfg=tp_overlap_cfg,
            overlap_param_gather_with_optimizer_step=True,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=22,
        )
    )
    recipe.trainer.callbacks.append(run.Config(GarbageCollectionCallback, gc_interval_train=100, gc_interval_val=100))
    recipe.trainer.strategy.cross_entropy_fusion_impl = "te"
    return recipe


# LLAMA3 405B Recipe
@run.cli.factory(target=llm.pretrain)
def cloudai_llama3_405b_recipe() -> run.Partial:
    recipe = run.Partial(
        llm.pretrain,
        model=run.Config(LlamaModel, config=Llama31Config405B()),
        data=run.Config(
            MockDataModule,
            seq_length=8192,
            micro_batch_size=1,
            global_batch_size=8,
            tokenizer=null_tokenizer(vocab_size=128256),
        ),
        trainer=run.Config(
            nl.Trainer,
            devices=8,
            num_nodes=1,
            accelerator="gpu",
            max_steps=10,
            limit_test_batches=50,
            limit_val_batches=32,
            log_every_n_steps=10,
            accumulate_grad_batches=1,
            plugins=run.Config(
                nl.MegatronMixedPrecision,
                precision="bf16-mixed",
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                autocast_enabled=False,
                grad_reduce_in_fp32=False,
            ),
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=8,
                pipeline_model_parallel_size=1,
                context_parallel_size=2,
                virtual_pipeline_model_parallel_size=8,
                sequence_parallel=True,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=None,
                pipeline_dtype=torch.bfloat16,
                gradient_as_bucket_view=True,
                ckpt_async_save=True,
                ckpt_parallel_load=True,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=False,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                ),
            ),
            num_sanity_val_steps=0,
            use_distributed_sampler=False,
            val_check_interval=1000,
            max_epochs=10,
            callbacks=[
                timing_callback(),
            ],
        ),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=0.0003,
                bf16=True,
                use_precision_aware_optimizer=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                weight_decay=0.1,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-05,
                clip_grad=1.0,
            ),
            lr_scheduler=run.Config(
                CosineAnnealingScheduler,
                warmup_steps=2000,
                constant_steps=0,
                min_lr=2.9999999999999997e-05,
            ),
        ),
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
    )
    recipe.model.config.expert_tensor_parallel_size = None
    recipe.model.config.seq_length = 8192

    tp_overlap_cfg, tp_comm_overlap = get_tp_overlap_config()
    megatron_comm_overlap_callback = run.Config(
        MegatronCommOverlapCallback,
        tp_comm_overlap=tp_comm_overlap,
        tp_comm_overlap_cfg=tp_overlap_cfg,
        overlap_param_gather_with_optimizer_step=True,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=50,
    )

    enable_fsdp = os.getenv("CLOUDAI_ENABLE_FSDP", "0") == "1"
    disable_tp_commd_overlap = os.getenv("CLOUDAI_DISABLE_TP_COMM_OVERLAP", "0") == "1"
    if enable_fsdp:
        recipe.trainer.limit_val_batches = 0
        recipe.model.config.init_model_with_meta_device = True
        recipe.trainer.strategy.fsdp = "megatron"
        recipe.trainer.strategy.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        recipe.trainer.strategy.ddp.average_in_collective = False
        recipe.trainer.strategy.ddp.keep_fp8_transpose_cache_when_using_custom_fsdp = False
        recipe.model.config.gradient_accumulation_fusion = False
        megatron_comm_overlap_callback.defer_embedding_wgrad_compute = False
        megatron_comm_overlap_callback.wgrad_deferral_limit = 50
        megatron_comm_overlap_callback.overlap_param_gather_with_optimizer_step = False

        if disable_tp_commd_overlap:
            megatron_comm_overlap_callback.tp_comm_overlap = False

    recompute_layers = int(os.getenv("CLOUDAI_RECOMPUTE_LAYERS", "0"))
    if recompute_layers > 0:
        recipe.model.config.recompute_granularity = "full"
        recipe.model.config.recompute_method = "block"
        recipe.model.config.recompute_num_layers = recompute_layers

    activation_offload_layers = int(os.getenv("CLOUDAI_ACTIVATION_OFFLOAD_LAYERS", "0"))
    if activation_offload_layers > 0:
        recipe.model.config.cpu_offloading = True
        recipe.model.config.cpu_offloading_weights = False
        recipe.model.config.cpu_offloading_num_layers = activation_offload_layers

    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="llama3",
        )
    )
    recipe.trainer.callbacks.append(megatron_comm_overlap_callback)
    recipe.trainer.callbacks.append(run.Config(GarbageCollectionCallback, gc_interval_train=100, gc_interval_val=100))
    recipe.trainer.strategy.account_for_embedding_in_pipeline_split = True
    recipe.trainer.strategy.account_for_loss_in_pipeline_split = True
    recipe.model.tokenizer = recipe.data.tokenizer
    recipe.trainer.strategy.cross_entropy_fusion_impl = "te"
    recipe.model.config.cross_entropy_fusion_impl = "te"

    if os.getenv("CLOUDAI_GPU_TYPE") in ["b200", "gb200"] and os.getenv("CLOUDAI_GPU_DTYPE") == "fp8":
        print("Info: use_precision_aware_optimizer is set to False for fp8 on b200/gb200 GPUs.")
        recipe.optim.config.use_precision_aware_optimizer = False
    return recipe


# NEMOTRON3 8B Recipe
@run.cli.factory(target=llm.pretrain)
def cloudai_nemotron3_8b_recipe() -> run.Partial:
    recipe = run.Partial(
        llm.pretrain,
        model=run.Config(nemotron3_8b_recipe(performance_mode=True)),
        data=run.Config(
            MockDataModule,
            seq_length=2048,
            micro_batch_size=4,
            global_batch_size=8,
            tokenizer=null_tokenizer(vocab_size=256000),
        ),
        trainer=run.Config(
            nl.Trainer,
            devices=8,
            num_nodes=1,
            accelerator="gpu",
            max_steps=10,
            limit_test_batches=50,
            limit_val_batches=32,
            log_every_n_steps=10,
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                sequence_parallel=False,
                pipeline_dtype=torch.bfloat16,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                ),
            ),
            num_sanity_val_steps=0,
            val_check_interval=1000,
            max_epochs=10,
        ),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                weight_decay=0,
            ),
        ),
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
    )
    recipe.model.config.vocab_size = 256000
    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="nemotron",
        )
    )
    set_enable_cuda_graphs_params(recipe)
    recipe.trainer.strategy.cross_entropy_fusion_impl = "te"
    return recipe


# NEMOTRON4 15B Recipe
@run.cli.factory(target=llm.pretrain)
def cloudai_nemotron4_15b_recipe() -> run.Partial:
    recipe = run.Partial(
        llm.pretrain,
        model=run.Config(NemotronModel, config=Nemotron4Config15B()),
        data=run.Config(
            MockDataModule,
            seq_length=4096,
            micro_batch_size=1,
            global_batch_size=8,
            tokenizer=null_tokenizer(vocab_size=256000),
        ),
        trainer=run.Config(
            nl.Trainer,
            devices=8,
            num_nodes=2,
            accelerator="gpu",
            max_steps=10,
            limit_test_batches=50,
            limit_val_batches=32,
            log_every_n_steps=10,
            use_distributed_sampler=False,
            val_check_interval=150,
            plugins=run.Config(
                nl.MegatronMixedPrecision,
                autocast_enabled=False,
                grad_reduce_in_fp32=False,
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                precision="bf16-mixed",
            ),
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                virtual_pipeline_model_parallel_size=None,
                sequence_parallel=True,
                pipeline_dtype=None,
                gradient_as_bucket_view=True,
                ckpt_async_save=True,
                ckpt_include_optimizer=True,
                ckpt_parallel_load=True,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                ),
            ),
            num_sanity_val_steps=0,
            max_epochs=10,
            callbacks=[timing_callback()],
        ),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                lr=1e-4,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
                weight_decay=0,
            ),
        ),
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
    )
    recipe.model.config.vocab_size = 256000
    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="nemotron",
        )
    )
    recipe.trainer.strategy.cross_entropy_fusion_impl = "te"
    set_enable_cuda_graphs_params(recipe)
    return recipe


# NEMOTRON4 340B Recipe
@run.cli.factory(target=llm.pretrain)
def cloudai_nemotron4_340b_recipe() -> run.Partial:
    recipe = run.Partial(
        llm.pretrain,
        model=run.Config(NemotronModel, config=Nemotron4Config340B()),
        data=run.Config(
            MockDataModule,
            seq_length=4096,
            micro_batch_size=1,
            global_batch_size=8,
            tokenizer=null_tokenizer(vocab_size=256000),
        ),
        trainer=run.Config(
            nl.Trainer,
            devices=8,
            num_nodes=32,
            accelerator="gpu",
            max_steps=10,
            limit_test_batches=32,
            limit_val_batches=0,
            log_every_n_steps=10,
            use_distributed_sampler=False,
            plugins=run.Config(
                nl.MegatronMixedPrecision,
                autocast_enabled=False,
                grad_reduce_in_fp32=False,
                params_dtype=torch.bfloat16,
                pipeline_dtype=torch.bfloat16,
                precision="bf16-mixed",
            ),
            strategy=run.Config(
                nl.MegatronStrategy,
                tensor_model_parallel_size=8,
                pipeline_model_parallel_size=8,
                context_parallel_size=2,
                virtual_pipeline_model_parallel_size=12,
                expert_model_parallel_size=1,
                expert_tensor_parallel_size=None,
                sequence_parallel=True,
                pipeline_dtype=torch.bfloat16,
                gradient_as_bucket_view=True,
                ckpt_async_save=True,
                ckpt_include_optimizer=True,
                ckpt_parallel_load=True,
                ddp=run.Config(
                    DistributedDataParallelConfig,
                    check_for_nan_in_grad=True,
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    average_in_collective=True,
                ),
                cross_entropy_fusion_impl="te",
            ),
            num_sanity_val_steps=0,
            val_check_interval=500,
            max_epochs=10,
            callbacks=[
                run.Config(
                    MegatronCommOverlapCallback,
                    tp_comm_overlap=True,
                    overlap_grad_reduce=True,
                    overlap_param_gather=True,
                    overlap_param_gather_with_optimizer_step=True,
                    defer_embedding_wgrad_compute=True,
                    wgrad_deferral_limit=22,
                ),
                timing_callback(),
            ],
        ),
        optim=run.Config(
            nl.MegatronOptimizerModule,
            config=run.Config(
                OptimizerConfig,
                optimizer="adam",
                lr=0.0001,
                weight_decay=0.1,
                fp16=False,
                bf16=True,
                use_precision_aware_optimizer=True,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1e-05,
                use_distributed_optimizer=True,
                clip_grad=1.0,
                params_dtype=torch.bfloat16,
            ),
            lr_scheduler=run.Config(
                CosineAnnealingScheduler,
                warmup_steps=500,
                constant_steps=0,
                min_lr=1e-5,
            ),
        ),
        resume=run.Config(
            nl.AutoResume,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
            resume_past_end=True,
        ),
        log=default_log(),
    )
    recipe.model.config.vocab_size = 256000
    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="nemotron",
        )
    )
    recipe.trainer.callbacks.append(run.Config(GarbageCollectionCallback, gc_interval_train=100, gc_interval_val=100))
    recipe.trainer.strategy.cross_entropy_fusion_impl = "te"
    recipe.model.config.cross_entropy_fusion_impl = "te"
    set_enable_cuda_graphs_params(recipe)

    if os.getenv("CLOUDAI_GPU_TYPE") in ["b200", "gb200"] and os.getenv("CLOUDAI_GPU_DTYPE") == "fp8":
        recipe.optim.config.use_precision_aware_optimizer = False

    return recipe


# Qwen3 30B Recipe
@run.cli.factory(target=llm.pretrain)
def cloudai_qwen3_30b_a3b_recipe() -> run.Partial:
    recipe = qwen3_30b_a3b_pretrain_recipe()

    recipe.log = default_log()

    if not hasattr(recipe.trainer, "callbacks") or recipe.trainer.callbacks is None:
        recipe.trainer.callbacks = []

    recipe.trainer.callbacks.append(
        run.Config(
            FLOPsMeasurementCallback,
            model_config=recipe.model.config,
            data_config=recipe.data,
            model_name="qwen3",
        )
    )

    recipe.trainer.callbacks.append(run.Config(MegatronTokenDropCallback))
    recipe.trainer.callbacks.append(run.Config(MegatronCommOverlapCallback, tp_comm_overlap=True))

    recipe.model.config.cross_entropy_fusion_impl = "te"
    recipe.model.config.cross_entropy_loss_fusion = True
    recipe.model.config.apply_rope_fusion = True
    recipe.model.config.moe_permute_fusion = True
    recipe.model.config.bias_dropout_fusion = True
    recipe.model.config.bias_activation_fusion = True

    recipe.model.config.recompute_granularity = None
    recipe.model.config.recompute_method = None
    recipe.model.config.recompute_num_layers = None

    if os.getenv("CLOUDAI_GPU_TYPE") in ("gb200", "b200"):
        set_enable_cuda_graphs_params(recipe)

    return recipe


if __name__ == "__main__":
    mode = os.getenv("CLOUDAI_NEMO_TASK")

    supported_recipes = [
        "cloudai_llama3_8b_recipe",
        "cloudai_llama3_70b_recipe",
        "cloudai_llama3_405b_recipe",
        "cloudai_nemotron3_8b_recipe",
        "cloudai_nemotron4_15b_recipe",
        "cloudai_nemotron4_340b_recipe",
        "cloudai_deepseek_v3_recipe",
        "cloudai_qwen3_30b_a3b_recipe",
    ]

    recipe_name = os.getenv("CLOUDAI_NEMO_RECIPE")

    if recipe_name not in supported_recipes:
        print(
            (
                f"Warning: Using Default Recipe '{recipe_name}'. "
                "Advanced CLI features that use ForwardRefs are not supported using in Nemo-Run CLI yet."
            )
        )
    if mode == "pretrain":
        run.cli.main(fn=llm.pretrain)
    elif mode == "finetune":
        run.cli.main(fn=llm.finetune)
    else:
        raise ValueError(f"Unknown mode {mode}")
