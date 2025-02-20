import os
import nemo_run as run
import torch

from nemo.collections import llm

from nemo.collections.llm.gpt.model.llama import Llama3Config8B, LlamaModel
from lightning.pytorch.loggers import WandbLogger
from nemo.lightning.pytorch.callbacks.nsys import NsysCallback
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.common.tokenizers.huggingface import AutoTokenizer
import logging
from nemo import lightning as nl

from nemo.lightning.pytorch.callbacks.garbage_collection import GarbageCollectionCallback
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.collections.llm.gpt.data.mock import MockDataModule
from nemo.lightning.run.plugins import NsysPlugin


@run.cli.factory
@run.autoconvert
def hf_tokenizer() -> run.Config[AutoTokenizer]:
    """
    HuggingFace tokenizer.

    Args:
        model_name (str): corresponds to HuggingFace-AutoTokenizer's 'pretrained_model_name_or_path' input argument.
                For more details please refer to-
                huggingface.co/docs/transformers/v4.47.1/en/model_doc/auto#transformers.AutoTokenizer
    """
    model_name = "meta-llama/Meta-Llama-3-8B"
 
    return run.Config(
        AutoTokenizer,
        pretrained_model_name=model_name,
        use_fast=True,
    )

@run.cli.factory
@run.autoconvert
def timing_callback() -> run.Config[TimingCallback]:
    return run.Config(
        TimingCallback
    )
@run.cli.factory
@run.autoconvert
def garbage_collection_callback()->run.Config[GarbageCollectionCallback]:
    return run.Config(
        GarbageCollectionCallback,
        gc_interval_train=100,
        gc_interval_val=100
    )
@run.cli.factory
@run.autoconvert
def nsys_callbacks() -> run.Config[NsysCallback]:
    start_step = 5
    end_step = 10
    return run.Config(
        NsysCallback,
        start_step=start_step,
        end_step=end_step,
    )

@run.cli.factory
@run.autoconvert
def comms_overlap_callback()->run.Config[MegatronCommOverlapCallback]:
    return run.Config(
        MegatronCommOverlapCallback,
        overlap_param_gather_with_optimizer_step=False,
    )

@run.cli.factory
@run.autoconvert
def combined_callbacks() -> list:
    return [
            nsys_callbacks(),
            timing_callback(),
            comms_overlap_callback(),
            garbage_collection_callback(),
        ]

@run.cli.factory(target=llm.pretrain)
@run.autoconvert
def base_pretrain_config(args) -> run.Partial:
    """Base Pretraining Config"""
    recipe = run.Partial(
        llm.pretrain,
        model=run.Config(
            LlamaModel,
            config=run.Config(Llama3Config8B),
        ),
        data=run.Config(
            MockDataModule,
            seq_length=2048,
            micro_batch_size=4,
            global_batch_size=8,
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
            callbacks = [run.Config(
                NsysCallback,
                start_step=5,
                end_step=6,
            )],
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
        log=nl.NeMoLogger(wandb=(WandbLogger() if "WANDB_API_KEY" in os.environ else None)),
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
    logging.info(f"Partial configuration: {recipe}")
    return recipe


if __name__ == "__main__":
    run.plugins = [NsysPlugin(start_step=5, end_step=10)]
    run.cli.main(fn=llm.pretrain)