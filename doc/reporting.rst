Reporting
=========

This chapter describes the reporting system in CloudAI. In this chapter, we will cover the following topics:

- :ref:`Overview <overview>`
- :ref:`General Flow <general-flow>`
- :ref:`Enabling, Disabling and Configuring Reports <enabling-disabling-and-configuring-reports>`
- :ref:`Reporting Registration <reporting-registration>`
- :ref:`Reporting Configuration Implementation <reporting-configuration-implementation>`

.. _overview:

**Overview**

CloudAI has two reporting levels:

- per-test (per each case in a test scenario)
- per-scenario (per each test scenario)

All reports are generated after the test scenario is completed as part of the main CloudAI process. For Slurm, this means that the login node is used to generate reports.

Per-test reports are linked to a particular workload type (e.g. ``NcclTest``). All per-test reports are implemented as part of the ``per_test`` scenario report and can be enabled or disabled via a single configuration option; see :ref:`enabling-disabling-and-configuring-reports`.

To list all available reports, users can use ``cloudai list-reports``. Use verbose output to also print report configurations.


.. _general-flow:

General Flow
------------

- All reports should be registered via ``Registry()`` (``.add_report()`` or ``.add_scenario_report()``)
- Scenario reports are configurable via system config (Slurm-only for now) and scenario config
- Configuration in a scenario config has the highest priority. Next, system config is checked. Then it defaults to report config from the registry
- Finally, the report is generated (or not) according to this final config


.. _enabling-disabling-and-configuring-reports:

Enabling, Disabling and Configuring Reports
-------------------------------------------

.. note::

   Only scenario-level reports can be configured.

Enabling or disabling a report needs to be done in the system configuration:

.. code-block:: toml

   [reports]
   per_test = { enable = false }
   status = { enable = true }

.. _reporting-registration:

Reporting Registration
----------------------

Report registration is done via ``Registry`` class:

.. code-block:: python

   Registry().add_scenario_report("per_test", PerTestReporter, ReportConfig(enable=True))

.. _reporting-configuration-implementation:

Reporting Configuration Implementation
---------------------------------------

Each report can define its own configuration, which is constructed and passed as an argument to ``Registry.add_scenario_report``.
The ``reports`` field is parsed during TOML reading and the respective Pydantic model is created for it.

For example, a custom report configuration can be defined as follows:

.. code-block:: python

   class CustomReportConfig(ReportConfig):
       greeting: str

.. code-block:: python

   Registry().add_scenario_report("custom", CustomReport, CustomReportConfig(greeting="default value"))

And it can be used in a test scenario as follows:

.. code-block:: toml

   [reports]
   custom = { enable = true, greeting = "Hello, world!" }
