Add a reporter
==============

Turn workload output into metrics or publish a scenario-level artifact.

Start with the reporter nearest your workload in ``src/cloudai/workloads`` or ``src/cloudai/report_generator``.
Import public interfaces through ``cloudai.core``. A workload reporter and a scenario reporter have different
registration contracts; select the level that owns the data you need.

.. _reporting-registration:

Reporting Registration
----------------------

Report registration is done via ``Registry`` class:

.. code-block:: python

   Registry().add_scenario_report("per_test", PerTestReporter, ReportConfig(enable=True))

.. _reporting-configuration-implementation:

Reporting Configuration Implementation
--------------------------------------

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


Verify successful, failed, and missing-metric cases. Return ``METRIC_ERROR`` for a failed metric,
using identity checks rather than treating every numeric ``-1.0`` as an error.
See :doc:`/reference/agents` for metric failure behavior.
