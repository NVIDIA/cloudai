Test definitions
================

A test definition selects a workload and its arguments. Keep it inline for one scenario; use a separate file when several scenarios share it.

Minimal definition
------------------

.. code-block:: toml

   name = "all-reduce"
   description = "NCCL collective communication"
   test_template_name = "NcclTest"

   [cmd_args]
   docker_image_url = "nvcr.io#nvidia/pytorch:25.12-py3"
   subtest_name = "all_reduce_perf_mpi"

Save this as ``tests/all-reduce.toml``. A scenario can refer to it with ``test_name = "all-reduce"``;
pass ``--tests-dir tests`` to the CLI. See :doc:`/guides/configuration` for the inline alternative.

Common settings
---------------

* ``name`` identifies the definition selected by a scenario.
* ``test_template_name`` selects the registered workload implementation.
* ``cmd_args`` contains workload-specific parameters; use its page in :doc:`/workloads/index` for exact fields.
* ``extra_env_vars`` supplies workload environment variables.
* ``extra_container_mounts`` exposes host paths to container workloads where supported.

Allocation and execution controls belong to the scenario case. List-valued parameters may define search dimensions;
consult :doc:`agents` for exclusions. Workload schemas are authoritative for which additional arguments they accept.
