Systems
========

Look up scheduler values and TOML fields for execution systems. Start with :doc:`guides/environment`
for setup instructions, then use this page to adapt resource and path settings.
Check :doc:`reference/workload-support` for the workloads supported on each system.

.. csv-table::
   :header: "System", "Scheduler Value"
   :widths: 40, 30

   ":ref:`slurm-system`", "``slurm``"
   ":ref:`standalone-system`", "``standalone``"
   ":ref:`kubernetes-system`", "``kubernetes``"
   ":ref:`runai-system`", "``runai``"
   ":ref:`lsf-system`", "``lsf``"

.. _slurm-system:

Slurm
-----

.. autopydantic_model:: cloudai.systems.slurm.slurm_system.SlurmSystem
   :exclude-members: cmd_shell, group_allocated, supports_gpu_directives_cache

.. autopydantic_model:: cloudai.systems.slurm.slurm_system.SlurmPartition
   :exclude-members: slurm_nodes

.. autopydantic_model:: cloudai.systems.slurm.slurm_system.SlurmGroup

.. autopydantic_model:: cloudai.systems.slurm.slurm_system.DataRepositoryConfig

.. _kubernetes-system:

Kubernetes
----------

Use :doc:`guides/kubernetes` for cluster setup and network attachments.

.. autopydantic_model:: cloudai.systems.kubernetes.kubernetes_system.KubernetesSystem

.. _runai-system:

RunAI
-----

The RunAI integration supports NCCL.

.. toctree::
   :maxdepth: 1

   NCCL on RunAI <guides/runai>

.. autopydantic_model:: cloudai.systems.runai.runai_system.RunAISystem
   :exclude-members: nodes

.. autopydantic_model:: cloudai.systems.runai.runai_node.RunAINode

.. _lsf-system:

LSF
---

.. autopydantic_model:: cloudai.systems.lsf.lsf_system.LSFSystem
   :exclude-members: cmd_shell

.. autopydantic_model:: cloudai.systems.lsf.lsf_system.LSFQueue
   :exclude-members: lsf_nodes

.. autopydantic_model:: cloudai.systems.lsf.lsf_system.LSFGroup

.. _standalone-system:

Standalone
----------

.. autopydantic_model:: cloudai.systems.standalone.standalone_system.StandaloneSystem
   :exclude-members: cmd_shell
