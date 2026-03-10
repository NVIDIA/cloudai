Systems
=======

This section lists all systems supported by CloudAI. The attributes shown for each system can be set in TOML configuration files.

.. csv-table::
   :header: "System", "Scheduler Value"
   :widths: 40, 30

   ":ref:`slurm-system`", "``slurm``"
   ":ref:`kubernetes-system`", "``kubernetes``"
   ":ref:`runai-system`", "``runai``"
   ":ref:`lsf-system`", "``lsf``"
   ":ref:`standalone-system`", "``standalone``"

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

.. autopydantic_model:: cloudai.systems.kubernetes.kubernetes_system.KubernetesSystem

.. _runai-system:

RunAI
-----

.. autopydantic_model:: cloudai.systems.runai.runai_system.RunAISystem
   :exclude-members: nodes

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
