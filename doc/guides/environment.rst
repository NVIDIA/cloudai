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


Kubernetes
----------

Use :doc:`kubernetes` to configure cluster access and networking, then run :doc:`Dynamo </workloads/ai_dynamo>`
or :doc:`NCCL </workloads/nccl>`. CloudAI generates the workload's Kubernetes resources and collects results.

Check :doc:`/reference/workload-support` before choosing a system. Configuration fields for all systems are
in :doc:`/systems`.
