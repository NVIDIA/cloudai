Troubleshooting
===============

Find the failed stage before changing the experiment. Start with the error message and the configuration used for that run.

.. list-table::
   :header-rows: 1

   * - Symptom
     - First check
     - Continue
   * - Configuration fails to load
     - File paths, workload name, field types, and referenced test definitions.
     - :doc:`guides/configuration` and :doc:`guides/preview`
   * - Prerequisite installation fails
     - Image or package availability, authentication, disk space, and cache path.
     - :doc:`workloads/workloads_requirements_installation`
   * - Job remains queued
     - Scheduler allocation, requested resources, and partition access.
     - :doc:`guides/environment`
   * - Workload exits unsuccessfully
     - Generated command, ``stdout.txt``, and ``stderr.txt`` for that execution.
     - Workload guide in :doc:`workloads/index`
   * - Metrics or reports are missing
     - Workload completion, expected output, parser inputs, and report configuration.
     - :doc:`reporting`
   * - Search selects an unexpected result
     - Objective direction, valid rewards, missing metrics, and compared conditions.
     - :doc:`optimization/objectives`

Inspect the execution
---------------------

CloudAI writes ``debug.log`` in the invocation directory unless ``--log-file`` selects another path.
Find the generated command and the execution's output directory there. On Slurm, inspect the generated sbatch script
and scheduler state. A pending allocation is different from a workload failure.

Reproduce narrowly
------------------

Use the same configuration with ``dry-run`` to inspect command generation. If you rerun the generated workload command,
it executes real work and may request resources. Reduce the scenario to the failing case while retaining its required
inputs and environment. Change one suspected cause at a time.

Report an issue
---------------

Use the `issue template <https://github.com/NVIDIA/cloudai/issues/new/choose>`_ with the CloudAI revision, invocation,
minimal configuration, expected result, and relevant logs. Remove credentials and internal hostnames before posting.
