Prepare model weights
=====================

Choose the model in the workload configuration before allocating compute. Use a small public model to verify the execution path.

Start small
-----------

The :doc:`/workloads/vllm` starter uses ``Qwen/Qwen3-0.6B``. Run it through the
:ref:`inference example <inference-with-vllm>` after preparing Slurm. Ensure the workload can reach its model source.

Use existing weights
--------------------

For vLLM, set ``cmd_args.model`` to the container-visible model directory and add the host directory to
``extra_container_mounts``. Follow :doc:`/workloads/vllm` for the local-model example.
For training, use the dataset and tokenizer settings described by the selected workload.

Access and storage
------------------

Authenticate with the model provider when the selected model requires it. Supply credentials through the environment
required by the workload; do not store tokens in scenario files. The Megatron Bridge starter requires ``HF_TOKEN``.
Allow space for the selected revision and verify the download using the provider's metadata before submitting jobs.
Model size, format, and access requirements depend on the selected artifact.

For larger models, first confirm device memory, parallelism, and shared-storage access on the intended nodes.
Keep the model revision and traffic or training inputs fixed when comparing configurations.
