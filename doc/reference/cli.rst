CLI reference
=============

Choose a command for validation, execution, prerequisite installation, or report regeneration.
For a complete first run, follow :doc:`/Tutorial`.

CloudAI Modes Usage Examples
----------------------------

The following are the global options for the ``cloudai`` command:

- ``--log-file <path>``: specify a file to log output; by default ``debug.log`` in the current directory is used. Contains log entries of level ``DEBUG`` and higher
- ``--log-level <level>``: specify logging level for standard output; default is ``INFO``

Available Modes
~~~~~~~~~~~~~~~

- :ref:`run`
- :ref:`dry-run`
- :ref:`generate-report`
- :ref:`install`
- :ref:`uninstall`
- :ref:`list`
- :ref:`verify-configs`

Usage Examples
~~~~~~~~~~~~~~

.. _run:

run
^^^

This mode runs workloads. It automatically installs prerequisites if they are not met.

.. code-block:: bash

   cloudai run\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --test-scenario conf/experimental/quickstart/nccl.toml

.. _dry-run:

dry-run
^^^^^^^

This mode generates workload commands without executing the workload. This is useful for verifying configurations and testing experiment setups.

.. code-block:: bash

   cloudai dry-run\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --test-scenario conf/experimental/quickstart/nccl.toml

.. _generate-report:

generate-report
^^^^^^^^^^^^^^^

This mode generates reports under the scenario directory. It automatically runs as part of the ``run`` mode after experiments are completed.

.. code-block:: bash

   cloudai generate-report\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --test-scenario conf/experimental/quickstart/nccl.toml\
       --result-dir /path/to/result_directory

Generated artifacts depend on the scenario contents:

- ``<scenario>.html`` summarizes scenario status.
- ``<scenario>-dse-report.html`` describes configuration searches when the scenario contains DSE test cases.
- ``<dse-case>/<iteration>/<dse-case>.toml`` contains the best discovered test configuration for a DSE case.
- Workload and custom reporters may produce additional artifacts.

The :doc:`/reporting` guide describes ``experiment.json``, which records execution status and workload metrics
independently of reporter settings.

.. _install:

install
^^^^^^^

This mode installs test prerequisites. For more details, refer to the :doc:`/workloads/workloads_requirements_installation` guide. It automatically runs as part of the ``run`` mode if prerequisites are not met.

.. code-block:: bash

   cloudai install\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --test-scenario conf/experimental/quickstart/nccl.toml

.. _uninstall:

uninstall
^^^^^^^^^

This mode is the opposite of the install mode. This mode removes installed test prerequisites.

.. code-block:: bash

   cloudai uninstall\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --test-scenario conf/experimental/quickstart/nccl.toml

.. _list:

list
^^^^

This mode lists internal components available within CloudAI.

.. code-block:: bash

   cloudai list <component_type>

.. _verify-configs:

verify-configs
^^^^^^^^^^^^^^

This mode verifies the correctness of system, test, and test scenario configuration files.

.. code-block:: bash

   # verify all at once
   cloudai verify-configs conf

   # verify a single file
   cloudai verify-configs conf/common/system/example_slurm_cluster.toml

   # verify all scenarios using specific folder with Test TOMLs
   cloudai verify-configs --tests-dir conf/release/spcx/l40s/test conf/release/spcx/l40s/test_scenario
