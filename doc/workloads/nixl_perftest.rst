NIXL CTPerf
===========

This workload (`test_template_name` is ``NixlPerftest``) runs NIXL performance testing suite for comprehensive network performance evaluation.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_nixl_perftest_test"
   description = "Example NIXL Perftest test"
   test_template_name = "NixlPerftest"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   subtest = "sequential-ct-perftest"
   num_user_requests = 1
   batch_size = 1
   num_prefill_nodes = 1
   num_decode_nodes = 1
   prefill_tp = 4
   decode_tp = 4
   isl_mean = 10000
   isl_scale = 3000
   model = "deepseek-r1-distill-llama-70b"

Test Scenario example:

.. code-block:: toml

   name = "nixl-perftest-test"

   [[Tests]]
   id = "perftest.1"
   num_nodes = 2
   time_limit = "00:20:00"

   test_name = "my_nixl_perftest_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "nixl-perftest-test"

   [[Tests]]
   id = "perftest.1"
   num_nodes = 2
   time_limit = "00:20:00"

   name = "my_nixl_perftest_test"
   description = "Example NIXL Perftest test"
   test_template_name = "NixlPerftest"

     [Tests.cmd_args]
     docker_image_url = "<docker container url here>"
     subtest = "sequential-ct-perftest"
     num_user_requests = 100
     batch_size = 1
     num_prefill_nodes = 1
     num_decode_nodes = 1
     prefill_tp = 8
     decode_tp = 8
     model = "deepseek-r1-distill-llama-70b"

     [Tests.extra_env_vars]
     CUDA_VISIBLE_DEVICES = "$SLURM_LOCALID"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_perftest.nixl_perftest.NixlPerftestCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_perftest.nixl_perftest.NixlPerftestTestDefinition
   :members:
   :show-inheritance:
