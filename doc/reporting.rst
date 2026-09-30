Reporting
=========

This chapter describes the reporting system in CloudAI. In this chapter, we will cover the following topics:

- :ref:`Overview <overview>`
- :ref:`General Flow <general-flow>`
- :ref:`Enabling, Disabling and Configuring Reports <enabling-disabling-and-configuring-reports>`
- :ref:`Reporting Registration <reporting-registration>`
- :ref:`Reporting Configuration Implementation <reporting-configuration-implementation>`
- :doc:`Reports <reports>`

.. toctree::
   :hidden:

   reports

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
   junit = { enable = true }

The ``junit`` scenario reporter is disabled by default. When enabled, it writes ``junit.xml`` in the scenario results
directory. It emits one test case for every regular test iteration and every DSE step, including pass/fail status,
failure details, scheduler duration when available, and the contents of ``stdout.txt`` and ``stderr.txt``. The artifact
can be consumed directly by Jenkins, GitLab, GitHub Actions, and other CI systems that support JUnit XML.

Speed-of-Light comparisons
--------------------------

CloudAI can compare NCCL and NIXL Bench measurements with configured Speed-of-Light (SOL) targets. Targets are
validated while the system or scenario TOML is parsed. A target without ``match`` is the default; a matching target
with more dimensions takes precedence:

.. code-block:: toml

   [[sol.bandwidth]]
   value = 100.0 # GB/s

   [[sol.bandwidth]]
   value = 120.0 # GB/s
   match = { operation = "write", size_bytes = 1048576 }

   [[sol.latency]]
   value = 8.0 # us

NCCL targets can distinguish the collective operation and placement:

.. code-block:: toml

   [[sol.bandwidth]]
   value = 250.0 # GB/s
   match = { operation = "all_reduce", placement = "out_of_place", bandwidth_basis = "bus" }

   [[sol.bandwidth]]
   value = 300.0 # GB/s
   match = { operation = "all_reduce", placement = "in_place", bandwidth_basis = "bus" }

A test case can replace the targets for a metric inherited from the scenario or system:

.. code-block:: toml

   [[Tests]]
   id = "nixl-case"
   test_name = "nixl-bench"

   [[Tests.sol.bandwidth]]
   value = 120.0

The precedence is test case, then scenario, then system. NCCL and NIXL comparison v2 reports include measured, SOL,
and percentage-of-SOL columns and draw a shared SOL curve when every compared run resolves the same targets.

.. _llm-serving-metric-observations:

LLM serving metric observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

vLLM and SGLang expose structured measurements through ``TestDefinition.metric_observations(system, test_run)``.
``cloudai.metrics.assess_test_run_metrics`` compares these observations with the run's configured SOL targets.

.. list-table:: Serving metrics
   :header-rows: 1

   * - Metric key
     - Unit
     - Preferred direction
   * - ``request_throughput``
     - requests/s
     - Higher
   * - ``output_token_throughput``
     - tokens/s
     - Higher
   * - ``ttft`` (time to first token)
     - ms
     - Lower
   * - ``tpot`` (time per output token)
     - ms
     - Lower
   * - ``accuracy``
     - ratio (0–1)
     - Higher

All observations include ``backend`` (``vllm`` or ``sglang``) and ``model`` dimensions. Benchmark observations also
include ``max_concurrency`` from the result file; TTFT and TPOT have a ``statistic`` dimension with ``mean``,
``median``, or ``p99``. Accuracy comes from semantic evaluation when ``semantic_eval_cmd_args`` is configured and
does not carry benchmark concurrency or latency statistics.

Request and output-token throughput are emitted only when their respective ``request_throughput`` and
``output_throughput`` fields are present. Missing or invalid benchmark results and results with no completed requests
produce no benchmark observations. Non-finite measurements are omitted. Semantic accuracy can be emitted
independently when an evaluation result is available.

For example, a system or scenario can define these targets:

.. code-block:: toml

   [[sol.output_token_throughput]]
   value = 5000.0 # tokens/s
   match = { model = "Qwen/Qwen3-8B", max_concurrency = 16 }

   [[sol.ttft]]
   value = 100.0 # ms
   match = { backend = "sglang", statistic = "p99" }

These observations leave the existing scalar metric names and comparison reports unchanged. In particular, the
legacy ``throughput`` metric uses output tokens/s for vLLM and requests/s for SGLang.

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
