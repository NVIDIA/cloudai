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


Controlling the Number of GPUs
-------------------------------
The number of GPUs can be controlled using the options below, listed from lowest to highest priority:
1. ``gpus_per_node`` system property (scalar value)
2. ``CUDA_VISIBLE_DEVICES`` environment variable (comma-separated list of GPU IDs)
3. ``gpu_ids`` command argument for ``prefill`` and ``decode`` configurations (comma-separated list of GPU IDs). If disaggregated mode is used (``prefill`` is set), both ``prefill`` and ``decode`` should define ``gpu_ids``, or none of them should set it.


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

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.vllm.vllm.VllmTestDefinition
   :members:
   :show-inheritance:
