NCCL
====

This workload (`test_template_name` is ``NcclTest``) allows users to execute NCCL benchmarks within the CloudAI framework.

Usage Example
-------------

Test TOML example:

.. code-block:: toml

   name = "my_nccl_test"
   description = "Example NCCL test"
   test_template_name = "NcclTest"

   [cmd_args]
   docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"

Test Scenario example:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "nccl.1"
   num_nodes = 1
   time_limit = "00:05:00"

   test_name = "my_nccl_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "nccl.1"
   num_nodes = 1
   time_limit = "00:05:00"

   name = "my_nccl_test"
   description = "Example NCCL test"
   test_template_name = "NcclTest"

     [Tests.cmd_args]
     docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"
     subtest_name = "all_reduce_perf_mpi"
     iters = 100

API Documentation
---------------------------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nccl_test.nccl.NCCLCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nccl_test.nccl.NCCLTestDefinition
   :members:
   :show-inheritance:
