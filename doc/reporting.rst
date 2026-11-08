Read results and generate reports
=================================

Open the results directory printed by CloudAI. Start with execution status, then inspect the measurements needed for your decision.

Find the evidence
-----------------

* ``experiment.json`` records cases, execution status, metrics, and selected search results.
* ``stdout.txt`` and ``stderr.txt`` explain what happened within a workload execution.
* Workload and scenario reports present measurements for inspection and comparison.

Compare only results with compatible model, workload, resources, input sizes, units, and measurement conditions.
For a failed or incomplete run, follow :doc:`troubleshooting` before interpreting its measurements.
Use :doc:`reference/cli` to regenerate reports from saved output, :doc:`pareto-frontiers` for trade-offs,
and :doc:`automation/publish` to publish artifacts.

.. _reporting-overview:

**Overview**

CloudAI has two reporting levels:

- per-test (per each case in a test scenario)
- per-scenario (per each test scenario)

All reports are generated after the test scenario is completed as part of the main CloudAI process. For Slurm, this means that the login node is used to generate reports.

Per-test reports are linked to a particular workload type (e.g. ``NcclTest``). All per-test reports are implemented as part of the ``per_test`` scenario report and can be enabled or disabled via a single configuration option; see :ref:`enabling-disabling-and-configuring-reports`.

To list all available reports, users can use ``cloudai list-reports``. Use verbose output to also print report configurations.


Unified Experiment Output
-------------------------

CloudAI writes ``experiment.json`` in each scenario's results directory.

The file contains scenario details, system name, the username of the account running CloudAI in ``user``, and test cases
listed under ``tests``. Older files without ``user`` are loaded with an empty string. For Slurm runs, the system name is
the cluster reported by Slurm. Both Standalone and Slurm execution records are stored in each test case's ``runs`` list. Each
record represents an iteration or DSE step and includes its number, process or Slurm job ID, status, timing, result path,
and workload metrics.

If username lookup fails, CloudAI logs a warning and records an empty ``user`` without interrupting execution.

Timestamps use UTC; durations use seconds. Unknown timestamps are ``null``. A status of ``unknown`` indicates the outcome
could not be determined. Dry runs include scenario and test-case details without launching workloads.

Live ``runs`` status tracking:

- Standalone jobs are ``running`` from submission.
- Slurm jobs (excluding single-sbatch) start as ``pending`` while waiting for allocation, then transition to ``running``
  once allocated and executing.
- Single-sbatch Slurm run records remain ``pending`` until the allocation finishes and their final statuses are recorded.

Metrics are sourced from ``TestDefinition.metric_observations()``, and are not affected by reporter settings.

MegatronRun exposes ``iteration_time`` (milliseconds, lower is better) and ``tflops_per_gpu``
(TFLOP/s/GPU, higher is better). Each observation is the mean of the last ten iteration metric lines in
``stdout.txt``, or all available lines when fewer than ten are present, matching the per-test CSV report.
No observations are emitted when iteration metric lines are missing.

When a test case is executed successfully, ``tests[].metrics`` contains that execution's metrics. For DSE, it contains
metrics from the successful step with the highest valid reward. The search space, selected step, and configuration appear
in ``tests[].dse``.

NCCL DSE Example
~~~~~~~~~~~~~~~~

This example comes from a Slurm NCCL all-reduce run on one node with eight H100 GPUs. DSE tried ``Ring`` and ``Tree``
in two steps. ``Ring`` (step 1) won using inverse latency as the reward, so its measurements appear in ``tests[].metrics``.

Identifiers and paths are anonymized. To keep the example small, each metrics list includes only out-of-place latency
and bus bandwidth for 1 MiB messages. Measurement values, timing, and DSE selection are unchanged.

:download:`Download experiment.json <examples/nccl-dse/experiment.json>`.

.. literalinclude:: examples/nccl-dse/experiment.json
   :language: json

Each file update replaces the previous snapshot atomically. Metric extraction or write errors produce warnings without
affecting execution.


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

Previous section links
----------------------

.. raw:: html

   <span id="reporting"></span>

Reporting: :doc:`/automation/publish`.

.. raw:: html

   <span id="reporting-registration"></span>

Reporting Registration: :doc:`/extend/reporter`.

.. raw:: html

   <span id="reporting-configuration-implementation"></span>

Reporting Configuration Implementation: :doc:`/extend/reporter`.

.. raw:: html

   <span id="uploading-results-to-object-storage"></span>

Uploading Results to Object Storage: :doc:`/automation/publish`.

.. raw:: html

   <span id="environment-variables"></span>

Environment variables: :doc:`/automation/publish`.
