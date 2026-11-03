Installation and CLI
====================

CloudAI officially targets Python 3.14 while remaining compatible with Python 3.10 and newer.

Use uv 0.13.4 or newer to install and run CloudAI. For pip-based installation, use pip 26.2.2 or newer.
These versions include certificate trust updates for the eMudhra/emSign CA compromise.

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloudai.git
   cd cloudai
   uv run cloudai --help

For a first benchmark, return to :ref:`getting-started`. See :doc:`systems` and
:doc:`workloads/workloads_requirements_installation` for environment setup.

pip-based Installation
----------------------

See the preferred Python version in the ``.python-version`` file and make sure you have it installed
(For installation, see :ref:`install-custom-python-version`). Python 3.10 and newer remain supported.
Follow these steps:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloudai.git
   cd cloudai
   python3.14 -m venv venv
   source venv/bin/activate
   pip install -e .

.. note::

   When upgrading an existing installation, also update the host's ``ca-certificates`` package and the
   certificate bundles and installers in existing workload environments and container images.
   Updating CloudAI does not update these environments.


.. _install-custom-python-version:

Custom Python Version Installation
----------------------------------

If your system Python version is not supported, you can install a custom version using the `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ tool:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv venv --seed  # picks the python version from .python-version
                   # --seed installs pip and setuptools
   source .venv/bin/activate

Key Concepts
------------

CloudAI separates where an experiment runs from what it executes:

- **System**: The execution environment, including the scheduler, available resources, and shared environment settings.
- **Test definition**: The workload and its command arguments, environment variables, and configuration choices.
- **Scenario**: The test cases to execute, with resource requirements and dependencies where supported.

See :doc:`USER_GUIDE` for configuration details and :doc:`Tutorial` for configuration examples.

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
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

.. _dry-run:

dry-run
^^^^^^^

This mode simulates running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups.

.. code-block:: bash

   cloudai dry-run\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

.. _generate-report:

generate-report
^^^^^^^^^^^^^^^

This mode generates reports under the scenario directory. It automatically runs as part of the ``run`` mode after experiments are completed.

.. code-block:: bash

   cloudai generate-report\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml\
       --result-dir /path/to/result_directory

Generated artifacts depend on the scenario contents:

- ``<scenario>.html`` summarizes scenario status.
- ``<scenario>-dse-report.html`` describes configuration searches when the scenario contains DSE test cases.
- ``<dse-case>/<iteration>/<dse-case>.toml`` contains the best discovered test configuration for a DSE case.
- Workload and custom reporters may produce additional artifacts.

The :doc:`reporting` guide describes ``experiment.json``, which records execution status and workload metrics
independently of reporter settings.

.. _install:

install
^^^^^^^

This mode installs test prerequisites. For more details, refer to the :doc:`workloads/workloads_requirements_installation` guide. It automatically runs as part of the ``run`` mode if prerequisites are not met.

.. code-block:: bash

   cloudai install\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

.. _uninstall:

uninstall
^^^^^^^^^

This mode is the opposite of the install mode. This mode removes installed test prerequisites.

.. code-block:: bash

   cloudai uninstall\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

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
