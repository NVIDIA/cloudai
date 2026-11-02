.. _cloudai-benchmark-framework:

CloudAI
========

**Benchmark and optimize AI systems.**

CloudAI is an open-source framework for benchmarking and optimizing AI systems. Run training, inference,
communication, and storage workloads through one experiment workflow. Use built-in grid search or connect custom
optimization agents to explore configurations and identify the best results observed for your objective.

:ref:`Get started <getting-started>` · :ref:`Choose a workflow <choose-a-workflow>` ·
:doc:`Workload compatibility <workloads/index>`

Overview
--------

.. topic:: Run different benchmarks

   Use a common CLI, Python API, and configuration model for workloads such as NCCL, NeMo, vLLM, SGLang, and Fio.

.. topic:: Search configurations

   Define configuration choices and a performance objective. Use grid search for small spaces or integrate your own
   search policy.

.. topic:: Compare and reuse results

   Inspect performance trade-offs, read structured experiment results, and reuse the best configuration discovered
   during a search.

.. toctree::
   :hidden:

   Tutorial
   USER_GUIDE
   api
   reporting
   systems
   workloads/index
   DEV
   troubleshooting

.. _choose-a-workflow:

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - I want to…
     - Start here
   * - Measure communication or storage performance
     - :doc:`NCCL <workloads/nccl>`, :doc:`NIXL <workloads/nixl_bench>`, or :doc:`Fio <workloads/fio>`
   * - Benchmark training configurations
     - :doc:`Megatron Bridge <workloads/megatron_bridge>` or :doc:`NeMo Run <workloads/nemo_run>`
   * - Compare inference configurations
     - :doc:`vLLM <workloads/vllm>`, :doc:`SGLang <workloads/sglang>`, and :doc:`latency–throughput trade-offs <pareto-frontiers>`
   * - Explore a configuration space
     - :ref:`configuration-search` and `agent configuration <USER_GUIDE.html#agents-configuration>`_
   * - Automate experiments and consume their results
     - :doc:`Python API <api>` and `experiment output <reporting.html#unified-experiment-output>`_

Workload availability depends on the execution environment. Check the :doc:`support matrix <workloads/index>`
before selecting a workload.

.. _configuration-search:

Configuration search
--------------------

CloudAI connects configuration choices, workload execution, and measured results in a design space exploration
(DSE) workflow. An optimization agent is the policy that selects configurations to evaluate and receives workload
measurements as feedback.

1. **Establish a baseline.** Measure the workload with fixed settings on your target system.
2. **Define the search space and objective.** Choose the settings to vary and the metric or reward to compare.
3. **Evaluate configurations.** Use the built-in ``grid_search`` agent or integrate a custom optimization agent
   through the public agent interface.
4. **Inspect and reuse the result.** Review the search history and best observed configuration in the DSE report.
   Use :doc:`Pareto frontiers <pareto-frontiers>` to compare measured trade-offs between metrics.

.. admonition:: Plan around your compute budget

   The built-in grid-search agent evaluates every combination in the configured search space. Estimate the number
   and cost of evaluations before launching, and keep the initial search small. Custom agents can implement adaptive
   selection and stopping policies through the public agent interface.

See `agent configuration <USER_GUIDE.html#agents-configuration>`_ for search settings and
:ref:`generate-report` for the resulting report and selected test configuration.

.. _getting-started:

Getting Started
---------------

CloudAI officially targets Python 3.14 while remaining compatible with Python 3.10 and newer.

Use uv 0.13.4 or newer to install and run CloudAI. For pip-based installation, use pip 26.2.2 or newer.
These versions include certificate trust updates for the eMudhra/emSign CA compromise.

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloudai.git
   cd cloudai
   uv run cloudai --help

.. note::

   Choose a workload from the table above, configure :doc:`where it will run <systems>`, and prepare its
   :doc:`prerequisites <workloads/workloads_requirements_installation>` before executing an experiment.

Use :ref:`dry-run` to inspect generated workload commands before execution. After an experiment, open its HTML
reports or read ``experiment.json`` in the results directory. See :doc:`reporting` for result interpretation and
:doc:`api` for programmatic access.

pip-based Installation
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
-----------------------------

The following are the global options for the ``cloudai`` command:

- ``--log-file <path>``: specify a file to log output; by default ``debug.log`` in the current directory is used. Contains log entries of level ``DEBUG`` and higher
- ``--log-level <level>``: specify logging level for standard output; default is ``INFO``

Available Modes
~~~~~~~~~~~~~~~~

- :ref:`run`
- :ref:`dry-run`
- :ref:`generate-report`
- :ref:`install`
- :ref:`uninstall`
- :ref:`list`
- :ref:`verify-configs`

Usage Examples
~~~~~~~~~~~~~~~

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

.. _external-materials-notice:

Notice and Disclaimer
---------------------

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.
Those retrieved materials are not distributed with this software and are governed solely by separate terms,
conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable
terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved
materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The
author makes no representations or warranties regarding any retrieved materials, and assumes no liability for
any losses, damages, liabilities or legal consequences from your use or inability to use this software or any
retrieved materials. Use this software and the retrieved materials at your own risk.

Documentation Update History
-----------------------------

+----------+-----------------+--------------------------------------------------+
| Version  | Date            | Description                                      |
+==========+=================+==================================================+
| v1.5.0   | April xx, 2026  | Initial release of this documentation version.   |
+----------+-----------------+--------------------------------------------------+
