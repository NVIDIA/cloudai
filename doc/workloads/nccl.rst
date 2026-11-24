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

Standalone MPI
--------------

With a standalone system, load the site's HPC-X environment before invoking
CloudAI. Use this test configuration instead of specifying a container:

.. code-block:: toml

   name = "nccl_mpi"
   description = "Bare-metal MPI bandwidth benchmark"
   test_template_name = "NcclTest"

   [standalone_mpi]
   launcher = "mpirun"
   binary_dir = "/shared/nccl-tests/build"
   ranks_per_node = 4
   launcher_args = ["--bind-to", "none"]

   [cmd_args]
   subtest_name = "all_reduce_perf"
   ngpus = 1

Set the system's ``scheduler = "standalone"``. In the scenario's ``[[Tests]]``
entry, set ``test_name = "nccl_mpi"``, ``num_nodes = 2`` and, for example,
``nodes = ["worker01", "worker02"]``. Use already assigned hosts: CloudAI
does not reserve them. All hosts must have access to the MPI-enabled binary
named by ``subtest_name`` and compatible libraries. Global and per-trial
environment settings are forwarded to MPI ranks. DSE uses the same NCCL
stdout and metric extraction as the containerized workload.

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
