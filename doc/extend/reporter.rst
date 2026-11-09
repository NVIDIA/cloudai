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

Provide inference measurements
------------------------------

Implement ``InferenceReportProvider.inference_reports(system, tr)`` from
``cloudai.report_generator.inference_report`` to make a workload's inference measurements available to shared reports.
Return one ``InferenceReport`` per independent measurement, or an empty list when no results are available.

.. list-table::
   :header-rows: 1

   * - Report content
     - Contract
   * - ``InferenceConfig``
     - Resolved model, benchmark, load, hardware, and serving-role settings. Unknown optional settings remain ``None``.
   * - ``MetricObservation`` outputs
     - Metric values with units and dimensions. Independent measurements stay in separate reports even if metric names match.
   * - Serving roles
     - Instance counts and assigned nodes per instance; TP, PP, and DP describe parallelism within each instance.
   * - ``shared_nodes``
     - Overlapping prefill/decode placement. Combined prefill/decode is displayed as ``Serving``.

vLLM and SGLang return one report per execution. Dynamo returns separate reports for benchmark phases and resolves
worker counts from its configured topology and standard launcher defaults.

The :doc:`Pareto reporter </pareto-frontiers>` treats each report as a candidate point and applies ``match`` to its
metric observations. ``metric_observations()`` remains available and combines observations from these reports.

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
