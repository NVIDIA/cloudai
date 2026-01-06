Reporting
=========

This document describes the reporting system in CloudAI.

Overview
--------

CloudAI has two reporting levels:

- per-test (per each case in a test scenario)
- per-scenario (per each test scenario)

All reports are generated after the test scenario is completed as part of the main CloudAI process. For Slurm this means that the login node is used to generate reports.

Per-test reports are linked to a particular workload type (e.g. ``NcclTest``). All per-test reports are implemented as part of the ``per_test`` scenario report and can be enabled or disabled via a single configuration option; see :ref:`enable-disable-and-configure-reports`.

To list all available reports, users can use ``cloudai list-reports``. Use verbose output to also print report configurations.

Notes and General Flow
----------------------

1. All reports should be registered via ``Registry()`` (``.add_report()`` or ``.add_scenario_report()``).
2. Scenario reports are configurable via system config (Slurm-only for now) and scenario config.
3. Configuration in a scenario config has the highest priority. Next, system config is checked. Then it defaults to report config from the registry.
4. Then the report is generated (or not) according to this final config.


.. _enable-disable-and-configure-reports:

Enable, Disable and Configure Reports
-------------------------------------

.. note::

   Only scenario-level reports can be configured.

To enable or disable a report, users need to do it via system configuration:

.. code-block:: toml

   [reports]
   per_test = { enable = false }
   status = { enable = true }

Report Registration
-------------------

Report registration is done via ``Registry`` class:

.. code-block:: python

   Registry().add_scenario_report("per_test", PerTestReporter, ReportConfig(enable=True))

Report Configuration Implementation
-----------------------------------

Each report can define its own configuration, which is constructed and passed as an argument to ``Registry.add_scenario_report``. The ``reports`` field is parsed during TOML reading and the respective Pydantic model is created.

For example, we can define a custom report configuration:

.. code-block:: python

   class CustomReportConfig(ReportConfig):
       greeting: str

.. code-block:: python

   Registry().add_scenario_report("custom", CustomReport, CustomReportConfig(greeting="default value"))

And use it in a test scenario:

.. code-block:: toml

   [reports]
   custom = { enable = true, greeting = "Hello, world!" }

