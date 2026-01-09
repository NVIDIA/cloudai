DeepEP Benchmark
================

This workload (``test_template_name`` is ``DeepEP``) allows users to execute DeepEP (Deep Expert Parallelism) MoE (Mixture of Experts) benchmarks within the CloudAI framework.

Overview
--------

DeepEP is a benchmark for measuring the performance of MoE models with distributed expert parallelism. It supports:

- **Two operation modes**: Standard and Low-Latency
- **Multiple data types**: bfloat16 and FP8
- **Flexible network configurations**: With or without NVLink
- **Configurable model parameters**: Experts, tokens, hidden size, top-k
- **Performance profiling**: Kineto profiler support

Usage Example
-------------

Test TOML example (Standard Mode):

.. code-block:: toml

   name = "deepep_standard"
   description = "DeepEP MoE Benchmark - Standard Mode"
   test_template_name = "DeepEP"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   mode = "standard"
   tokens = 1024
   num_experts = 256
   num_topk = 8
   hidden_size = 7168
   data_type = "bfloat16"
   num_warmups = 20
   num_iterations = 50

Test TOML example (Low-Latency Mode):

.. code-block:: toml

   name = "deepep_low_latency"
   description = "DeepEP MoE Benchmark - Low Latency Mode"
   test_template_name = "DeepEP"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   mode = "low_latency"
   tokens = 128
   num_experts = 256
   num_topk = 1
   hidden_size = 7168
   data_type = "bfloat16"
   allow_nvlink_for_low_latency = false
   allow_mnnvl = false

Test Scenario example:

.. code-block:: toml

   name = "deepep-benchmark"

   [[Tests]]
   id = "Tests.1"
   test_name = "deepep_standard"
   num_nodes = 2
   time_limit = "00:30:00"

Test-in-Scenario example:

.. code-block:: toml

   name = "deepep-benchmark"

   [[Tests]]
   id = "Tests.1"
   num_nodes = 2
   time_limit = "00:30:00"

   name = "deepep_standard"
   description = "DeepEP MoE Benchmark"
   test_template_name = "DeepEP"

     [Tests.cmd_args]
     docker_image_url = "<docker container url here>"
     mode = "standard"
     tokens = 1024
     num_experts = 256
     num_topk = 8

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.deepep.deepep.DeepEPCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.deepep.deepep.DeepEPTestDefinition
   :members:
   :show-inheritance:

