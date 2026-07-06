SGLang
======

This workload (``test_template_name`` is ``sglang``) allows users to execute SGLang benchmarks within the CloudAI framework.

SGLang is a high-throughput and memory-efficient inference engine for LLMs. This workload supports both aggregated and disaggregated prefill/decode modes.

Usage Examples
--------------

Test and Scenario Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Workload-specific test definition sections, such as ``bench_cmd_args`` and ``semantic_eval_cmd_args``, are not
supported under ``[[Tests]]`` in a test scenario. Define them in a test definition TOML and reference that test with
``test_name`` when custom benchmark or semantic-evaluation arguments are needed.


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


Reporting
---------
After a run completes, CloudAI parses ``sglang-bench.jsonl`` and prints serving latency, successful prompt count,
completion rate, throughput, TPS per user, and TPS per GPU. If ``semantic_eval_cmd_args`` is configured, CloudAI also
reports semantic validation accuracy.

The reported metric (``default``) is throughput. Additional supported metrics are ``throughput``, ``tps-per-user``,
``tps-per-gpu``, and ``accuracy``.

CloudAI also provides the scenario-level ``sglang_comparison`` report. It compares SGLang test runs in the scenario and
uses ``bench_cmd_args`` values as comparison labels.


Readiness health checks
-----------------------
Healthcheck fields:

- ``healthcheck``: aggregated server and disaggregated router endpoint, default ``/v1/models``.
- ``serve_healthcheck``: optional override for serve, prefill, and decode servers.

If ``serve_healthcheck`` is omitted, disaggregated prefill/decode servers keep the legacy ``/health`` endpoint.


Control number of GPUs
----------------------
GPU selection priority, from lowest to highest:

1. ``gpus_per_node`` system property (scalar value)
2. ``decode.gpu_ids`` command argument in non-disaggregated mode when ``CUDA_VISIBLE_DEVICES`` is not set
3. ``CUDA_VISIBLE_DEVICES`` environment variable (comma-separated list of GPU IDs)
4. ``gpu_ids`` command argument for both ``prefill`` and ``decode`` configurations in disaggregated mode

In disaggregated mode, define both ``prefill.gpu_ids`` and ``decode.gpu_ids``, or omit both.


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
For non-disaggregated ``num_nodes > 1``, CloudAI starts one ``sglang.launch_server`` task per serving node with shared
``--dist-init-addr``, ``--nnodes``, and ``--node-rank "$SLURM_PROCID"``.

For disaggregated serving over more than two nodes, set explicit role sizes:

- ``prefill.num_nodes + decode.num_nodes`` must equal the test ``num_nodes``.
- CloudAI assigns contiguous node slices: prefill first, decode second.
- ``tp`` is total per role, not per node.
- ``CUDA_VISIBLE_DEVICES`` and ``gpu_ids`` are local GPU IDs on each serving node.

Example: four prefill nodes and four decode nodes, each with four visible GPUs:

.. code-block:: toml
   :caption: scenario.toml (multi-node disaggregated serving)

   [[Tests]]
   id = "sglang.pd_multi_node"
   num_nodes = 8
   test_template_name = "sglang"

   [Tests.cmd_args.prefill]
   num_nodes = 4
   tp = 16

   [Tests.cmd_args.decode]
   num_nodes = 4
   tp = 16

   [Tests.extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

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
