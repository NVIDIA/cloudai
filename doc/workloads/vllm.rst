vLLM
====

vLLM workload (``test_template_name`` is ``vllm``) allows users to execute vLLM benchmarks within the CloudAI framework.

vLLM is a high-throughput and memory-efficient inference engine for LLMs. This workload supports both aggregated and disaggregated prefill/decode modes.

Usage Examples
--------------

Test and Scenario Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml
   :caption: test.toml (test definition)

   name = "vllm_test"
   description = "Example vLLM test"
   test_template_name = "vllm"

   [cmd_args]
   docker_image_url = "nvcr.io#nvidia/ai-dynamo/vllm-runtime:0.7.0"
   model = "Qwen/Qwen3-0.6B"

   [bench_cmd_args]
   random_input_len = 16
   random_output_len = 128
   max_concurrency = 16
   num_prompts = 30

   [semantic_eval_cmd_args]
   entrypoint = "python3 /opt/vllm/tests/evals/gsm8k/gsm8k_eval.py"
   cli = "--host {host} --port {port} --num-questions 200 --save-results {output_path}/vllm-gsm8k.json"


.. code-block:: toml
   :caption: scenario.toml (scenario with one test)

   name = "vllm-benchmark"

   [[Tests]]
   id = "vllm.1"
   num_nodes = 1
   time_limit = "00:10:00"
   test_name = "vllm_test"

Test-in-Scenario example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml
   :caption: scenario.toml (separate test toml is not needed)

   name = "vllm-benchmark"

   [[Tests]]
   id = "vllm.1"
   num_nodes = 1
   time_limit = "00:10:00"

   name = "vllm_test"
   description = "Example vLLM test"
   test_template_name = "vllm"

   [Tests.cmd_args]
   docker_image_url = "nvcr.io#nvidia/ai-dynamo/vllm-runtime:0.7.0"
   model = "Qwen/Qwen3-0.6B"

   [Tests.bench_cmd_args]
   random_input_len = 16
   random_output_len = 128
   max_concurrency = 16
   num_prompts = 30


Semantic Validation
-------------------
To run GSM8K semantic validation after the serving benchmark, add ``semantic_eval_cmd_args``. CloudAI reports
``accuracy`` from the eval output, but does not enforce an accuracy threshold.

.. code-block:: toml
   :caption: test.toml (semantic validation)

   [semantic_eval_cmd_args]
   entrypoint = "python3 /opt/vllm/tests/evals/gsm8k/gsm8k_eval.py"
   cli = "--host {host} --port {port} --num-questions 200 --save-results {output_path}/vllm-gsm8k.json"

If the runtime image does not contain the eval script, mount a vLLM repository with existing ``git_repos`` support and
point ``entrypoint`` at the mounted path.

The ``cli`` string supports ``{model}``, ``{host}``, ``{port}``, ``{url}``, ``{output_path}``, and ``{result_dir}``
placeholders.


Controlling the Number of GPUs
-------------------------------
The number of GPUs can be controlled using the options below, listed from lowest to highest priority:
1. ``gpus_per_node`` system property (scalar value)
2. ``decode.gpu_ids`` command argument in non-disaggregated mode when ``CUDA_VISIBLE_DEVICES`` is not set
3. ``CUDA_VISIBLE_DEVICES`` environment variable (comma-separated list of GPU IDs)
4. ``gpu_ids`` command argument for both ``prefill`` and ``decode`` configurations in disaggregated mode

For backward compatibility, non-disaggregated configs that set both ``CUDA_VISIBLE_DEVICES`` and ``decode.gpu_ids`` use
``CUDA_VISIBLE_DEVICES``. In disaggregated mode (``prefill`` is set), both ``prefill`` and ``decode`` should define
``gpu_ids``, or none of them should set it.

Controlling Disaggregation
--------------------------
By default, vLLM will run without disaggregation as a single process. To enable disaggregation, one needs to set ``prefill`` configuration:

.. code-block:: toml
   :caption: test.toml (disaggregated prefill/decode)

   [cmd_args]
   docker_image_url = "nvcr.io#nvidia/ai-dynamo/vllm-runtime:0.7.0"
   model = "Qwen/Qwen3-0.6B"

   [cmd_args.prefill]

   [extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

The config above, will automatically split GPUs specified in ``CUDA_VISIBLE_DEVICES`` into two:
- The first half will be used for prefill
- The second half will be used for decode

For more control, users can specify the GPU IDs explicitly in ``prefill`` and ``decode`` configurations:

.. code-block:: toml
   :caption: test.toml (disaggregated prefill/decode)

   [cmd_args.prefill]
   gpu_ids = "0,1"

   [cmd_args.decode]
   gpu_ids = "2,3"

In this case ``CUDA_VISIBLE_DEVICES`` will be ignored and only the GPUs specified in ``gpu_ids`` will be used.


Multi-node serving
------------------
For non-disaggregated serving, set ``num_nodes`` on the test to more than one. CloudAI starts a Ray head on the first
allocated serving node, Ray workers on the remaining serving nodes, waits for the Ray cluster to reach the requested
size, and runs ``vllm serve`` with ``--distributed-executor-backend ray`` on the head node.
``tensor_parallel_size`` is the total tensor-parallel size across the Ray serving role. With two nodes and
``CUDA_VISIBLE_DEVICES = "0,1,2,3"`` on each node, set ``tensor_parallel_size = 8`` to use all eight visible GPUs.

.. code-block:: toml
   :caption: scenario.toml (multi-node aggregated serving)

   [[Tests]]
   id = "vllm.multi_node"
   num_nodes = 2
   test_template_name = "vllm"

   [Tests.cmd_args]
   docker_image_url = "nvcr.io/nvidia/vllm:latest"
   model = "Qwen/Qwen3-0.6B"

   [Tests.cmd_args.decode]
   tensor_parallel_size = 8

   [Tests.extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

For disaggregated prefill/decode serving, existing 1-node and 2-node behavior is preserved by default. To span more
than two nodes, set both role sizes explicitly. CloudAI assigns contiguous node slices to prefill and decode, creates a
separate Ray cluster for each role whose ``num_nodes`` is greater than one, and runs benchmark and semantic validation
from the prefill head node.
Role ``tensor_parallel_size`` values are total per Ray role, not per node. For example, ``num_nodes = 2`` with four
visible GPUs per node uses ``tensor_parallel_size = 8`` to consume all GPUs in that role.

.. code-block:: toml
   :caption: scenario.toml (multi-node disaggregated serving)

   [[Tests]]
   id = "vllm.pd_multi_node"
   num_nodes = 4
   test_template_name = "vllm"

   [Tests.cmd_args]
   docker_image_url = "nvcr.io/nvidia/vllm:latest"
   model = "Qwen/Qwen3-0.6B"

   [Tests.cmd_args.prefill]
   num_nodes = 2
   tensor_parallel_size = 8

   [Tests.cmd_args.decode]
   num_nodes = 2
   tensor_parallel_size = 8

   [Tests.extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

``CUDA_VISIBLE_DEVICES`` and ``gpu_ids`` are interpreted as local GPU IDs on each serving node, not as cluster-global GPU
IDs.


Readiness health checks
-----------------------
CloudAI waits for vLLM servers to become ready before starting the benchmark. The default vLLM server endpoint remains
``/healthcheck`` for backward compatibility with existing configs and runtime images. Generated Slurm scripts wait for
the configured endpoint exactly.

Use ``serve_healthcheck`` to override the readiness endpoint for the vLLM serve process, including prefill/decode server
processes in disaggregated mode. If ``serve_healthcheck`` is not set, aggregated serving uses ``healthcheck``.
Disaggregated prefill/decode serving keeps the legacy ``/health`` default.

In disaggregated mode, ``proxy_healthcheck`` controls the proxy/router readiness endpoint. Existing disaggregated
configs that set ``healthcheck`` and do not set ``proxy_healthcheck`` continue to use ``healthcheck`` for the
proxy/router check.

For custom runtime images with a different readiness path, set ``serve_healthcheck`` for vLLM server processes and,
when using disaggregated mode, ``proxy_healthcheck`` for the proxy/router.


Controlling ``proxy_script``
-----------------------------
``proxy_script`` is used to proxy the requests from the client to the prefill and decode instances. It is ignored for non-disaggregated mode. Default value can be found below.

It can be overridden by setting ``proxy_script`` by using the latest version of the script from vLLM repository:

.. code-block:: toml
   :caption: test_scenario.toml (override proxy_script)

   [[Tests.git_repos]]
   url = "https://github.com/vllm-project/vllm.git"
   commit = "main"
   mount_as = "/vllm_repo"

   [Tests.cmd_args]
   docker_image_url = "vllm/vllm-openai:v0.14.0-cu130"
   proxy_script = "/vllm_repo/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py"

In this case the proxy script will be mounted from the vLLM repository (cloned locally) as ``/vllm_repo`` and used for the test.


API Documentation
-----------------

vLLM Serve Arguments
~~~~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.vllm.vllm.VllmArgs
   :members:

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.vllm.vllm.VllmCmdArgs
   :members:
   :show-inheritance:

Benchmark Command Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.vllm.vllm.VllmBenchCmdArgs
   :members:
   :show-inheritance:

Semantic Eval Command Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.vllm.vllm.VllmSemanticEvalCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.vllm.vllm.VllmTestDefinition
   :members:
   :show-inheritance:
