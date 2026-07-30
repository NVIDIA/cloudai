MegatronBridge
==============

This workload (`test_template_name` is ``MegatronBridge``) submits training and finetuning tasks based on Megatron-Bridge framework.

.. note::

   This workload has a hard requirement for the HuggingFace Hub token. There are two options:

   - (recommended) define ``HF_TOKEN`` environment variable
   - set ``cmd_args.hf_token`` either in Test or Scenario config


Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "megatron_bridge_qwen_30b"
   description = "Megatron-Bridge run via CloudAI SlurmSystem for Qwen3 30B A3B"
   test_template_name = "MegatronBridge"

   [[git_repos]]
   url = "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
   commit = "v0.3.0"
   mount_as = "/opt/Megatron-Bridge"

   [cmd_args]
   gpu_type = "gb200"
   gpus_per_node = 8
   num_gpus = 8
   # Container can be an NGC/enroot URL (nvcr.io#...) or a local .sqsh path.
   container_image = "nvcr.io#nvidia/nemo:26.02.00"

   model_family_name = "qwen"
   model_recipe_name = "qwen3_30b_a3b"
   task = "pretrain"
   domain = "llm"
   compute_dtype = "fp8_mx"

Test Scenario example:

.. code-block:: toml

   name = "megatron_bridge_qwen_30b"

   [[Tests]]
   id = "megatron_bridge_qwen_30b"
   test_name = "megatron_bridge_qwen_30b"
   num_nodes = "2"

Test-in-Scenario example:

.. code-block:: toml

   name = "megatron-bridge-test"

   [[Tests]]
   id = "mbridge.1"
   num_nodes = 2
   time_limit = "00:30:00"

   name = "megatron_bridge_qwen_30b"
   description = "Megatron-Bridge run via CloudAI SlurmSystem for Qwen3 30B A3B"
   test_template_name = "MegatronBridge"

     [[Tests.git_repos]]
     url = "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
     commit = "v0.3.0"
     mount_as = "/opt/Megatron-Bridge"

     [Tests.cmd_args]
     container_image = "nvcr.io#nvidia/nemo:26.02.01"
     model_family_name = "qwen"
     model_recipe_name = "qwen3_30b_a3b"

     gpu_type = "gb200"
     gpus_per_node = 8
     num_gpus = 8

     task = "pretrain"
     domain = "llm"
     compute_dtype = "fp8_mx"

Chakra / Kineto Tracing
-----------------------

Disabling nsys
~~~~~~~~~~~~~~

The existing MegatronBridge TOMLs in ``conf/staging/`` have ``enable_nsys = true``.
To switch to Kineto/Chakra profiling, either:

- **Omit** ``enable_nsys`` from the TOML (it defaults to ``false``)
- **Or** explicitly set ``enable_nsys = false``

Both are equivalent. Do not set ``enable_nsys = true`` alongside Kineto flags — the
two profilers conflict and Megatron-Bridge will raise an assertion error.


Enabling Kineto Trace
~~~~~~~~~~~~~~~~~~~~~

Add the following to ``[extra_cmd_args]`` in your Test TOML. These are Hydra-style
overrides passed directly to the Megatron-Bridge training script:

.. code-block:: toml

   [extra_cmd_args]
   "profiling.use_pytorch_profiler" = "true"
   "profiling.profile_step_start" = "45"
   "profiling.profile_step_end" = "50"
   "profiling.profile_ranks" = "[0]"

This produces the following under the NemoRun experiment directory (``<cloudai_output>/experiments/<experiment_name>/<run_id>/<experiment_name>/``):

- ``torch_profile/rank-0.json.gz`` — Chrome-format kineto trace (always produced)
- ``tb_logs/`` — TensorBoard events


Enabling Chakra Trace
~~~~~~~~~~~~~~~~~~~~~

The Chakra trace is a PyTorch ``ExecutionTraceObserver`` recording — a separate file
from the kineto trace. Add ``pytorch_profiler_collect_chakra = true``:

.. code-block:: toml

   [extra_cmd_args]
   "profiling.use_pytorch_profiler" = "true"
   "profiling.profile_step_start" = "45"
   "profiling.profile_step_end" = "50"
   "profiling.profile_ranks" = "[0]"
   "profiling.pytorch_profiler_collect_chakra" = "true"

This produces an additional ``chakra/rank-0.json.gz`` in the same experiment directory.
The kineto trace (``torch_profile/rank-0.json.gz``) is always produced regardless of
this flag.


Reference TOML
~~~~~~~~~~~~~~

A ready-to-use TOML is provided at:

.. code-block:: text

   conf/experimental/megatron_bridge/test/gb200/megatron_bridge_nemotron_nano_pytorch_profiler.toml

This runs Nemotron 3 Nano 3B (8 GPUs, 60 steps) with Kineto and Chakra profiling at
steps 45–50 and is suitable for quickly validating the profiling setup.


Profiling Parameter Reference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Parameter
     - Type
     - Description
   * - ``profiling.use_pytorch_profiler``
     - bool
     - Enable PyTorch kineto profiler. Required for all Kineto/Chakra tracing.
   * - ``profiling.profile_step_start``
     - int
     - Training step at which profiling begins.
   * - ``profiling.profile_step_end``
     - int
     - Training step at which profiling ends and trace is exported.
   * - ``profiling.profile_ranks``
     - list[int]
     - GPU ranks to profile (e.g. ``[0]`` for rank 0 only).
   * - ``profiling.record_shapes``
     - bool
     - Record tensor shapes in the kineto trace. Expensive — off by default.
   * - ``profiling.record_memory_history``
     - bool
     - Record memory allocation history (produces ``memory_profile.pickle``). Expensive — off by default.
   * - ``profiling.memory_snapshot_path``
     - str
     - Filename for the memory snapshot pickle file.
   * - ``profiling.pytorch_profiler_collect_chakra``
     - bool
     - Collect Chakra ExecutionTraceObserver trace (``chakra/rank-N.json.gz``). The kineto trace is always produced; this adds an additional file.
   * - ``profiling.pytorch_profiler_collect_callstack``
     - bool
     - Include Python call stack in the kineto trace (increases trace size).
   * - ``enable_nsys``
     - bool
     - Enable Nsight Systems profiling. **Cannot be used with Kineto.** Default: ``false``.


API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.megatron_bridge.megatron_bridge.MegatronBridgeCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.megatron_bridge.megatron_bridge.MegatronBridgeTestDefinition
   :members:
   :show-inheritance:
