SGLang
======

This workload (``test_template_name`` is ``sglang``) executes SGLang serving benchmarks in CloudAI.

It supports both aggregated mode (single server) and prefill/decode disaggregated mode with an SGLang router.

Usage Example
-------------

.. code-block:: toml
   :caption: test.toml

   name = "sglang_test"
   description = "Example SGLang benchmark"
   test_template_name = "sglang"

   [cmd_args]
   docker_image_url = "docker.io/lmsysorg/sglang:dev"
   model = "Qwen/Qwen3-8B"
   port = 8000

   [cmd_args.decode]
   tp = 2

   [bench_cmd_args]
   num_prompts = 1000
   max_concurrency = 10
   random_input = 1024
   random_output = 1024

To enable disaggregated mode, set ``cmd_args.prefill``:

.. code-block:: toml
   :caption: disaggregated example

   [cmd_args.prefill]
   tp = 2

   [cmd_args.decode]
   tp = 1

   [extra_env_vars]
   CUDA_VISIBLE_DEVICES = "0,1,2,3"

This runs prefill/decode servers, starts the SGLang router, waits for health checks, and then launches
``sglang.bench_serving``.

API Documentation
-----------------

SGLang Serve Arguments
~~~~~~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.sglang.sglang.SglangArgs
   :members:

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.sglang.sglang.SglangCmdArgs
   :members:
   :show-inheritance:

Benchmark Command Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.sglang.sglang.SglangBenchCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.sglang.sglang.SglangTestDefinition
   :members:
   :show-inheritance:
