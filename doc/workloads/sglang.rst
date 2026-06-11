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

   [semantic_eval_cmd_args]
   entrypoint = "python3 -m sglang.test.run_eval"
   cli = "--host {host} --port {port} --eval-name gsm8k --num-examples 200 --num-threads 128 --model {model}"


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


Semantic Validation
-------------------
To run GSM8K semantic validation after the serving benchmark, add ``semantic_eval_cmd_args``. CloudAI reports
``accuracy`` from the eval output, but does not enforce an accuracy threshold.

.. code-block:: toml
   :caption: test.toml (semantic validation)

   [semantic_eval_cmd_args]
   entrypoint = "python3 -m sglang.test.run_eval"
   cli = "--host {host} --port {port} --eval-name gsm8k --num-examples 200 --num-threads 128 --model {model}"

For images that still use the legacy SGLang GSM8K runner, override the entrypoint and raw CLI:

.. code-block:: toml

   [semantic_eval_cmd_args]
   entrypoint = "python3 -m sglang.test.few_shot_gsm8k"
   cli = "--host {host} --port {port} --num-questions 200"

The ``cli`` string supports ``{model}``, ``{host}``, ``{port}``, ``{url}``, ``{output_path}``, and ``{result_dir}``
placeholders.


Readiness health checks
-----------------------
CloudAI waits for SGLang servers to become ready before starting the benchmark. The default SGLang readiness endpoint is
``/v1/models``. Set ``serve_healthcheck`` to override the endpoint used for aggregated serve processes and
disaggregated prefill/decode server processes. In disaggregated mode, the router readiness check uses ``healthcheck``.


Control number of GPUs
----------------------
The number of GPUs can be controlled using the options below, listed from lowest to highest priority:
1. ``gpus_per_node`` system property (scalar value)
2. ``decode.gpu_ids`` command argument in non-disaggregated mode when ``CUDA_VISIBLE_DEVICES`` is not set
3. ``CUDA_VISIBLE_DEVICES`` environment variable (comma-separated list of GPU IDs)
4. ``gpu_ids`` command argument for both ``prefill`` and ``decode`` configurations in disaggregated mode

For backward compatibility, non-disaggregated configs that set both ``CUDA_VISIBLE_DEVICES`` and ``decode.gpu_ids`` use
``CUDA_VISIBLE_DEVICES``. In disaggregated mode (``prefill`` is set), both ``prefill`` and ``decode`` should define
``gpu_ids``, or none of them should set it.


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

Multi-node serving
------------------
For non-disaggregated serving, set ``num_nodes`` on the test to more than one. CloudAI starts one
``sglang.launch_server`` task per serving node with a shared ``--dist-init-addr``, ``--nnodes``, and
``--node-rank "$SLURM_NODEID"``.
SGLang ``tp`` is the total tensor-parallel size for the distributed serving role. With two nodes and
``CUDA_VISIBLE_DEVICES = "0,1,2,3"`` on each node, set ``tp = 8`` to use all eight visible GPUs.

.. code-block:: toml
   :caption: scenario.toml (multi-node aggregated serving)

   [[Tests]]
   id = "sglang.multi_node"
   num_nodes = 2
   test_template_name = "sglang"

   [Tests.cmd_args]
   docker_image_url = "lmsysorg/sglang:dev-cu13"
   model = "Qwen/Qwen3-8B"

   [Tests.cmd_args.decode]
   tp = 8

   [Tests.extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

For disaggregated prefill/decode serving, existing 1-node and 2-node behavior is preserved by default. To span more
than two nodes, set both role sizes explicitly. CloudAI assigns contiguous node slices to prefill and decode and starts
one distributed SGLang launch per role with separate init ports. Benchmark and semantic validation run from the prefill
head node.
Role ``tp`` values are total per distributed role, not per node. For example, ``num_nodes = 2`` with four visible GPUs
per node uses ``tp = 8`` to consume all GPUs in that role.

.. code-block:: toml
   :caption: scenario.toml (multi-node disaggregated serving)

   [[Tests]]
   id = "sglang.pd_multi_node"
   num_nodes = 4
   test_template_name = "sglang"

   [Tests.cmd_args]
   docker_image_url = "lmsysorg/sglang:dev-cu13"
   model = "Qwen/Qwen3-8B"

   [Tests.cmd_args.prefill]
   num_nodes = 2
   tp = 8

   [Tests.cmd_args.decode]
   num_nodes = 2
   tp = 8

   [Tests.extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

``CUDA_VISIBLE_DEVICES`` and ``gpu_ids`` are interpreted as local GPU IDs on each serving node, not as cluster-global GPU
IDs.

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

Semantic Eval Command Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.sglang.sglang.SglangSemanticEvalCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.sglang.sglang.SglangTestDefinition
   :members:
   :show-inheritance:
