VAR="$(scontrol show hostname \"${SLURM_STEP_NODELIST}\" | head -n1)" \
__OUTPUT_DIR__/install/NeMo-Framework-Launcher__599ecfcbbd64fd2de02f2cc093b1610d73854022-venv/bin/python \
 __OUTPUT_DIR__/install/NeMo-Framework-Launcher__599ecfcbbd64fd2de02f2cc093b1610d73854022/launcher_scripts/main.py \
 cluster.gpus_per_node=8 \
 numa_mapping.enable=True \
 stages=["training"] \
 training.exp_manager.create_checkpoint_callback=False \
 training.model.data.data_impl=mock \
 training.model.data.data_prefix=[] \
 training.model.global_batch_size=128 \
 training.model.micro_batch_size=2 \
 training.model.pipeline_model_parallel_size=4 \
 training.model.tensor_model_parallel_size=4 \
 training.run.name=run \
 training.run.time_limit=3:00:00 \
 training.trainer.enable_checkpointing=False \
 training.trainer.log_every_n_steps=1 \
 training.trainer.max_steps=20 \
 training.trainer.val_check_interval=10 \
 training=gpt3/40b_improved \
 cluster.partition=main \
 training.trainer.num_nodes=1 \
 container=nvcr.io/nvidia/nemo:24.12.01 \
 cluster.job_name_prefix=test_account-cloudai.nemo: \
 base_results_dir=__OUTPUT_DIR__/output \
 launcher_scripts_path=__OUTPUT_DIR__/install/NeMo-Framework-Launcher__599ecfcbbd64fd2de02f2cc093b1610d73854022/launcher_scripts \
 +env_vars.VAR="$(scontrol show hostname \"${SLURM_STEP_NODELIST}\" | head -n1)"
