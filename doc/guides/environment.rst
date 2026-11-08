Set up your execution environment
=================================

Choose a system supported by the workload, then configure its resources and paths.

Standalone
----------

For local Fio execution, install ``fio`` and copy the system example:

.. code-block:: bash

   cp conf/common/system/standalone_system.toml standalone.toml

Use :doc:`/Tutorial` to verify this path. Standalone support varies by workload; check :doc:`/workloads/index`.

Slurm
--------

On the submit host, prepare :doc:`/workloads/workloads_requirements_installation`, then copy:

.. code-block:: bash

   cp conf/common/system/example_slurm_cluster.toml slurm.toml

Edit ``slurm.toml`` for your cluster:

* Set partition, node counts, and GPU counts.
* Make output, install, model, and input paths accessible on the execution nodes.
* Add container mounts for data outside the default paths.

Start with one case. Use :doc:`slurm` for allocation modes, scheduler arguments, and tracing.


Other schedulers
----------------

Use :doc:`runai` for RunAI credentials and configuration. The :doc:`/workloads/ai_dynamo` guide includes a Kubernetes
execution path. Consult :doc:`/systems` for exact system fields and :doc:`/workloads/index` for workload compatibility.
System registration alone does not imply every workload supports that scheduler.
