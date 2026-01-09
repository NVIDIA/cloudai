NIXL KVBench
============

This workload (`test_template_name` is ``NIXLKVBench``) runs NIXL KV-cache benchmarking for key-value store performance testing.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_nixl_kvbench_test"
   description = "Example NIXL KVBench test"
   test_template_name = "NIXLKVBench"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   model = "./examples/model_deepseek_r1.yaml"
   model_config = "./examples/block-tp1-pp16.yaml"
   backend = "POSIX"
   num_requests = 1
   source = "file"
   num_iter = 16
   page_size = 256
   filepath = "/data"

Test Scenario example:

.. code-block:: toml

   name = "nixl-kvbench-test"

   [[Tests]]
   id = "kvbench.1"
   num_nodes = 1
   time_limit = "00:10:00"

   test_name = "my_nixl_kvbench_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "nixl-kvbench-test"

   [[Tests]]
   id = "kvbench.1"
   num_nodes = 1
   time_limit = "00:10:00"

   name = "my_nixl_kvbench_test"
   description = "Example NIXL KVBench test"
   test_template_name = "NIXLKVBench"

     [Tests.cmd_args]
     docker_image_url = "<docker container url here>"
     backend = "UCX"
     source = "memory"
     op_type = "READ"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_kvbench.nixl_kvbench.NIXLKVBenchCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_kvbench.nixl_kvbench.NIXLKVBenchTestDefinition
   :members:
   :show-inheritance:
