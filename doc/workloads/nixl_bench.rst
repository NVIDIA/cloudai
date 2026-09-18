NIXL Bench
==========

This workload (`test_template_name` is ``NIXLBench``) runs NIXL benchmarking suite for network and interconnect performance testing.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_nixl_bench_test"
   description = "Example NIXL Bench test"
   test_template_name = "NIXLBench"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   path_to_benchmark = "/workspace/nixlbench/build/nixlbench"
   backend = "UCX"
   initiator_seg_type = "VRAM"
   target_seg_type = "VRAM"
   op_type = "READ"
   filepath = "/data"
   device_list = "11:F:/store0.bin"
   # one could also use <num>kb, <num>mb, <num>gb shortcuts
   total_buffer_size = 8000000000

Test Scenario example:

.. code-block:: toml

   name = "nixl-bench-test"

   [[Tests]]
   id = "bench.1"
   num_nodes = 1
   time_limit = "00:10:00"

   test_name = "my_nixl_bench_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "nixl-bench-test"

   [[Tests]]
   id = "bench.1"
   num_nodes = 1
   time_limit = "00:10:00"

   name = "my_nixl_bench_test"
   description = "Example NIXL Bench test"
   test_template_name = "NIXLBench"

     [Tests.cmd_args]
     docker_image_url = "<docker container url here>"
     path_to_benchmark = "/workspace/nixlbench/build/nixlbench"
     backend = "UCX"
     initiator_seg_type = "DRAM"
     target_seg_type = "DRAM"
     op_type = "WRITE"

Runtime Coordination
--------------------

NIXLBench uses ETCD by default. CloudAI starts ETCD from the benchmark image when
``etcd_image_url`` is omitted, or from the configured image.

To use NIXLBench's direct two-process ASIO runtime instead, set:

.. code-block:: toml

   runtime_type = "ASIO"

CloudAI resolves ``asio_address`` to the first allocated node by default and does not
install or start ETCD. ``asio_address`` and ``asio_port`` can be overridden explicitly.
ASIO requires exactly two NIXLBench processes. For UCX, a one-node test runs both
processes locally, while a two-node test runs one process on each node.

Storage backends can run without either runtime by using an empty ETCD endpoint:

.. code-block:: toml

   backend = "POSIX"
   etcd_endpoints = ""

This null-runtime mode is limited to storage backends and launches one NIXLBench process.

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchCmdArgs
   :members:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchTestDefinition
   :members:
   :show-inheritance:
