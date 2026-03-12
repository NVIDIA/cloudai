SGLang
======

This workload (``test_template_name`` is ``sglang``) allows users to execute SGLang benchmarks within the CloudAI framework.

SGLang is a high-throughput and memory-efficient inference engine for LLMs. This workload supports both aggregated and disaggregated prefill/decode modes.

Usage Examples
--------------

Test + Scenario example
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml
   :caption: test.toml (test definition)

   name = "sglang_test"
   description = "Example SGLang benchmark"
   test_template_name = "sglang"

   [cmd_args]
   docker_image_url = "lmsysorg/sglang:dev-cu13"
   model = "Qwen/Qwen3-8B"

   [bench_cmd_args]
   random_input = 16
   random_output = 128
   max_concurrency = 16
   num_prompts = 30


.. code-block:: toml
   :caption: scenario.toml (scenario with one test)

   name = "sglang-benchmark"

   [[Tests]]
   id = "sglang.1"
   num_nodes = 1
   time_limit = "00:10:00"
   test_name = "sglang_test"

Test-in-Scenario example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml
   :caption: scenario.toml (separate test toml is not needed)

   name = "sglang-benchmark"

   [[Tests]]
   id = "sglang.1"
   num_nodes = 1
   time_limit = "00:10:00"

   name = "sglang_test"
   description = "Example SGLang benchmark"
   test_template_name = "sglang"

   [Tests.cmd_args]
   docker_image_url = "lmsysorg/sglang:dev-cu13"
   model = "Qwen/Qwen3-8B"

   [Tests.bench_cmd_args]
   random_input = 16
   random_output = 128
   max_concurrency = 16
   num_prompts = 30


Control number of GPUs
----------------------
The number of GPUs can be controlled using the options below, listed from lowest to highest priority:
1. ``gpus_per_node`` system property (scalar value)
2. ``CUDA_VISIBLE_DEVICES`` environment variable (comma-separated list of GPU IDs)
3. ``gpu_ids`` command argument for ``prefill`` and ``decode`` configurations (comma-separated list of GPU IDs). If disaggregated mode is used (``prefill`` is set), both ``prefill`` and ``decode`` should define ``gpu_ids``, or none of them should set it.


Control disaggregation
----------------------
By default, SGLang will run without disaggregation as a single process. To enable disaggregation, one needs to set ``prefill`` configuration:

.. code-block:: toml
   :caption: test.toml (disaggregated prefill/decode)

   [cmd_args]
   docker_image_url = "lmsysorg/sglang:dev-cu13"
   model = "Qwen/Qwen3-8B"

   [cmd_args.prefill]

   [extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

The config above will automatically split GPUs specified in ``CUDA_VISIBLE_DEVICES`` into two halves, first half will be used for prefill and second half will be used for decode.

For more control, one can specify the GPU IDs explicitly in ``prefill`` and ``decode`` configurations:

.. code-block:: toml
   :caption: test.toml (disaggregated prefill/decode)

   [cmd_args.prefill]
   gpu_ids = "0,1"

   [cmd_args.decode]
   gpu_ids = "2,3"

In this case ``CUDA_VISIBLE_DEVICES`` will be ignored and only the GPUs specified in ``gpu_ids`` will be used.

API Documentation
-----------------

SGLang Serve Arguments
~~~~~~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.sglang.sglang.SglangArgs
   :members:

Command Arguments
~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.sglang.sglang.SglangCmdArgs
   :members:
   :show-inheritance:

Benchmark Command Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.sglang.sglang.SglangBenchCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.sglang.sglang.SglangTestDefinition
   :members:
   :show-inheritance:
