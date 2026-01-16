CloudAI Benchmark Framework
===========================

CloudAI benchmark framework aims to develop an industry standard benchmark focused on grading Data Center (DC) scale AI systems in the cloud. The primary motivation is to provide automated benchmarking on various systems.

Get Started
-----------

.. code-block:: bash

   git clone git@github.com:NVIDIA/cloudai.git
   cd cloudai
   uv run cloudai --help

.. note::

   For instructions for setting up access for ``enroot``, see :doc:`workloads_requirements_installation`.

``pip``-based Installation
--------------------------

See the required Python version in the ``.python-version`` file and make sure you have it installed (for installation, see :ref:`install-custom-python-version`). Follow these steps:

.. code-block:: bash

   git clone git@github.com:NVIDIA/cloudai.git
   cd cloudai
   python -m venv venv
   source venv/bin/activate
   pip install -e .


.. _install-custom-python-version:

Install Custom Python Version
-----------------------------

If your system Python version is not supported, you can install a custom version using the `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ tool:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv venv --seed  # picks the python version from .python-version
                   # --seed installs pip and setuptools
   source .venv/bin/activate

Key Concepts
------------

CloudAI operates on three main schemas:

- **System Schema**: Describes the system, including the scheduler type, node list, and global environment variables.
- **Test Schema**: An instance of a test template with custom arguments and environment variables.
- **Test Scenario Schema**: A set of tests with dependencies and additional descriptions about the test scenario.

These schemas enable CloudAI to be flexible and compatible with different systems and configurations.

CloudAI Modes Usage Examples
----------------------------

Global options for ``cloudai`` command:

- ``--log-file <path>``: specify a file to log output; by default ``debug.log`` in the current directory is used. Contains log entries of level ``DEBUG`` and higher.
- ``--log-level <level>``: specify logging level for standard output; default is ``INFO``.

.. _run:

run
~~~

This mode runs workloads. It automatically installs prerequisites if they are not met.

.. code-block:: bash

   cloudai run\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

.. _dry-run:

dry-run
~~~~~~~

This mode simulates running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups.

.. code-block:: bash

   cloudai dry-run\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

.. _generate-report:

generate-report
~~~~~~~~~~~~~~~

This mode generates reports under the scenario directory. It automatically runs as part of the ``run`` mode after experiments are completed.

.. code-block:: bash

   cloudai generate-report\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml\
       --result-dir /path/to/result_directory

.. _install:

install
~~~~~~~

This mode installs test prerequisites. For more details, refer to the :doc:`workloads_requirements_installation` guide. It automatically runs as part of the ``run`` mode if prerequisites are not met.

.. code-block:: bash

   cloudai install\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

.. _uninstall:

uninstall
~~~~~~~~~

The opposite to the install mode, this mode removes installed test prerequisites.

.. code-block:: bash

   cloudai uninstall\
       --system-config conf/common/system/example_slurm_cluster.toml\
       --tests-dir conf/common/test\
       --test-scenario conf/common/test_scenario/sleep.toml

.. _list:

list
~~~~

This mode lists internal components available within CloudAI.

.. code-block:: bash

   cloudai list <component_type>

.. _verify-configs:

verify-configs
~~~~~~~~~~~~~~

This mode verifies the correctness of system, test, and test scenario configuration files.

.. code-block:: bash

   # verify all at once
   cloudai verify-configs conf

   # verify a single file
   cloudai verify-configs conf/common/system/example_slurm_cluster.toml

   # verify all scenarios using specific folder with Test TOMLs
   cloudai verify-configs --tests-dir conf/release/spcx/l40s/test conf/release/spcx/l40s/test_scenario

CloudAI
-------

.. toctree::
   :maxdepth: 1

   USER_GUIDE
   reporting
   DEV
   workloads/index
   workloads_requirements_installation

