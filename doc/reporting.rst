Reporting
=========

This chapter describes the reporting system in CloudAI. In this chapter, we will cover the following topics:

- :ref:`Overview <overview>`
- :ref:`General Flow <general-flow>`
- :ref:`Enabling, Disabling and Configuring Reports <enabling-disabling-and-configuring-reports>`
- :ref:`Reporting Registration <reporting-registration>`
- :ref:`Reporting Configuration Implementation <reporting-configuration-implementation>`
- :ref:`Uploading Results to Object Storage <uploading-results-to-object-storage>`
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


Unified Experiment Output
-------------------------

CloudAI writes ``experiment.json`` in each scenario's results directory.

The file contains scenario details, system name, and test cases listed under ``tests``. For Slurm runs, the system name is
the cluster reported by Slurm. Both Standalone and Slurm execution records are stored in each test case's ``runs`` list. Each
record represents an iteration or DSE step and includes its number, process or Slurm job ID, status, timing, result path,
and workload metrics.

Timestamps use UTC; durations use seconds. Unknown timestamps are ``null``. A status of ``unknown`` indicates the outcome
could not be determined. Dry runs include scenario and test-case details without launching workloads.

For standalone runs and Slurm's default mode (one submission per run), CloudAI refreshes the file on submission,
completion, and an observed status transition. Status checks follow the system's ``monitor_interval``. Standalone runs
are ``running`` from launch; Slurm runs transition
from ``pending`` to ``running`` when the existing job checks report execution. Slurm start times and durations remain
``null`` until completion supplies accounting metadata. Running durations reflect the time of the last published snapshot. Finish times remain ``null`` until completion.
Snapshots replace the file atomically so readers can safely read it while the experiment is running.

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

.. _uploading-results-to-object-storage:

Uploading Results to Object Storage
------------------------------------

The ``s3`` scenario report publishes the scenario results directory to an
S3-compatible bucket. It is disabled by default, because shipping results off-box should
be a deliberate choice.

It is registered last, after ``tarball``, so it always observes the complete results
directory including every other report's output.

Install the optional dependency first:

.. code-block:: bash

   pip install 'cloudai[s3]'

Then enable it in a test scenario:

.. code-block:: toml

   [reports]
   s3 = { enable = true, bucket = "my-bucket", prefix = "cloudai/runs", upload_tarball = true }

Or, for Slurm systems, once per cluster in the system config:

.. code-block:: toml

   [reports]
   s3 = { enable = true, bucket = "my-bucket" }

Configuration options:

.. list-table::
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - ``bucket``
     - ``$CLOUDAI_S3_BUCKET``
     - Destination bucket. Required; the upload is skipped with a warning if unset.
   * - ``prefix``
     - ``$CLOUDAI_S3_PREFIX``
     - Key prefix. Objects are written under ``<prefix>/<system_name>/<results_dir_name>/``.
   * - ``endpoint_url``
     - ``$CLOUDAI_S3_ENDPOINT_URL``
     - Custom endpoint, for MinIO or other S3-compatible stores.
   * - ``region``
     - unset
     - AWS region. When unset, boto3 resolves it (e.g. ``AWS_DEFAULT_REGION``).
   * - ``upload_tree``
     - ``true``
     - Upload each file individually, preserving relative paths.
   * - ``upload_tarball``
     - ``false``
     - Also upload a ``.tgz`` of the whole directory. It is created if absent and rebuilt if older than the results.
   * - ``upload_concurrency``
     - ``8``
     - Number of files uploaded concurrently when ``upload_tree`` is enabled. Must be at least 1.

At least one of ``upload_tree`` and ``upload_tarball`` must be enabled; the config is rejected otherwise.

Environment variables
~~~~~~~~~~~~~~~~~~~~~

The destination can be supplied through three environment variables, so a cluster-wide
default does not have to be repeated in every scenario:

.. list-table::
   :header-rows: 1

   * - Variable
     - Sets option
     - Notes
   * - ``CLOUDAI_S3_BUCKET``
     - ``bucket``
     - Destination bucket.
   * - ``CLOUDAI_S3_PREFIX``
     - ``prefix``
     - Key prefix. Empty by default.
   * - ``CLOUDAI_S3_ENDPOINT_URL``
     - ``endpoint_url``
     - Custom endpoint, for MinIO or other S3-compatible stores. An empty value is treated as unset.

A value set in TOML takes precedence over the environment variable. The variables are read by
the ``cloudai`` process when the report configuration is loaded, so export them where
``cloudai`` runs:

.. code-block:: bash

   export CLOUDAI_S3_BUCKET=my-bucket
   export CLOUDAI_S3_PREFIX=cloudai/runs
   export CLOUDAI_S3_ENDPOINT_URL=http://localhost:9000   # only for MinIO or other S3-compatible stores

With these set, enabling the report only needs ``s3 = { enable = true }``.

**Credentials are never read from CloudAI configuration.** They are resolved by boto3's
standard chain: ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY``, ``~/.aws/credentials``,
or an instance/IAM role.

Because reports run inside a ``try``/``except``, an upload failure logs a warning and
leaves the run's exit status unchanged.

To upload a results directory from an earlier run, re-run the reports against it:

.. code-block:: bash

   cloudai generate-report --system-config <system.toml> --tests-dir <tests/> \
     --test-scenario <scenario.toml> --result-dir results/<scenario>_<timestamp>
