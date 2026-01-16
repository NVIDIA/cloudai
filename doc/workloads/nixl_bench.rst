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

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchTestDefinition
   :members:
   :show-inheritance:
