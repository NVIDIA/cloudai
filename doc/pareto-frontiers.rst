Pareto Frontiers
================

The ``pareto`` scenario report writes ``pareto.html`` in the scenario results directory. Each plot compares two
metrics. Successful iterations and sweep steps supply the points: one test case can contribute several points,
and several cases can contribute to one frontier. Each configured Dynamo AIPerf phase contributes separately.

Point details show serving instances, nodes per instance, and TP/PP/DP within each instance. Disaggregated
runs show separate prefill and decode layouts and whether they share nodes; combined serving uses the
``Serving`` label.

Usage Example
-------------

For :doc:`vLLM <workloads/vllm>`, :doc:`SGLang <workloads/sglang>`, or
:doc:`Dynamo <workloads/ai_dynamo>`, add this block to your scenario TOML:

.. code-block:: toml

   [reports.pareto]
   enable = true

   [[reports.pareto.plots]]
   name = "Streaming latency vs throughput"
   x = { metric = "tpot", match = { statistic = "p99" } }
   y = { metric = "output_token_throughput" }
   frontiers = [{ name = "All configurations" }]

This selects P99 time per output token on X and total output tokens/s on Y. CloudAI minimizes latency and
maximizes throughput using the metrics' default directions. Both axes increase normally.

``frontiers`` defines the comparison groups in the plot. Omitting ``cases`` or setting ``cases = []`` includes
all cases in the scenario. Use this example when all cases belong to the same comparison.

Available Metrics
-----------------

Use the metric keys below in either ``x.metric`` or ``y.metric``. Each axis must select exactly one measurement
for each point. ``match`` selects among measurements of the same metric, such as different statistics or
message sizes. Missing or ambiguous measurements are listed as exclusions in the report.

Both ``vllm`` and ``sglang`` expose these metrics:

.. list-table::
   :header-rows: 1
   :widths: 65 20 15

   * - Metric
     - Unit
     - Default direction
   * - ``ttft``
     - ms
     - minimize
   * - ``tpot``
     - ms
     - minimize
   * - ``output_token_throughput``
     - tokens/s
     - maximize
   * - ``request_throughput``
     - requests/s
     - maximize
   * - ``output_token_throughput_per_allocated_gpu``
     - tokens/s/GPU
     - maximize
   * - ``output_token_throughput_per_serving_gpu``
     - tokens/s/GPU
     - maximize
   * - ``accuracy``
     - ratio
     - maximize

``ttft`` measures the wait for the first token; ``tpot`` measures time per output token after the first token.
For either metric, select ``mean``, ``median``, or ``p99`` with ``match = { statistic = "p99" }``. The throughput
metrics have no statistic selector. ``accuracy`` is available when semantic validation is configured and its
result is present; see the workload's Semantic Validation section.

Dynamo exposes the same metrics except ``output_token_throughput_per_serving_gpu``. It reads each phase's
``profile_export_aiperf.json``, with CSV support for older AIPerf and GenAI-Perf results. Its ``tpot`` measurements
use Inter Token Latency. Missing statistics are omitted.

AIPerf JSON exports also provide these Dynamo metrics, all maximized by default:

.. list-table::
   :header-rows: 1
   :widths: 65 35

   * - Metric
     - Unit
   * - ``total_token_throughput``
     - tokens/s
   * - ``total_token_throughput_per_user``
     - tokens/s/user
   * - ``total_token_throughput_per_serving_gpu``
     - tokens/s/GPU

Total throughput includes input and output tokens, using AIPerf's ``total_token_throughput.avg``. Per-user
throughput divides it by the phase's configured ``args.users``, falling back to ``args.concurrency``. Per-serving-GPU
throughput divides it by the configured GPU count across prefill and decode, including worker replicas and
multi-node groups. SGLang attention DP shares TP devices. Omitted settings follow CloudAI's standard launcher
defaults; custom commands that change GPU placement are outside this calculation. Packing and hardware-derived
defaults use the system's ``gpus_per_node``, or ``CUDA_VISIBLE_DEVICES`` when configured for the test.

The two GPU-normalized metrics divide total output tokens/s by different counts:

- ``output_token_throughput_per_allocated_gpu`` uses the execution's ``num_nodes`` times the system's
  ``gpus_per_node``. It includes idle GPUs on those nodes. A larger single-sbatch reservation does not change
  the individual execution's count.
- ``output_token_throughput_per_serving_gpu`` uses the configured serving layout, summing prefill and decode
  GPUs when both roles are configured. It describes configured devices, not measured GPU utilization.

Choose the denominator that matches your comparison and use it consistently. A normalized metric is omitted
when its GPU count cannot be determined.

Examples and Best Practices
---------------------------

For inference, start with one of these comparisons:

- **First-token wait vs capacity:** ``ttft`` on X, ``output_token_throughput`` on Y.
- **Streaming latency vs serving efficiency:** ``tpot`` on X, ``output_token_throughput_per_serving_gpu`` on Y.
- **Streaming latency vs allocated GPU efficiency:** ``tpot`` on X,
  ``output_token_throughput_per_allocated_gpu`` on Y.
- **Dynamo total-token efficiency:** ``ttft`` on X, ``total_token_throughput_per_serving_gpu`` on Y.

Keep the model, input/output token lengths, precision, and traffic pattern comparable. Vary concurrency or
serving parameters to measure the tradeoff. Use P99 for tail latency, or select mean or median when that is
the statistic your comparison requires. Match the statistic and GPU denominator when comparing with another
benchmark report.

Multiple Plots
~~~~~~~~~~~~~~

Append another ``[[reports.pareto.plots]]`` block to generate a separate chart. For example, add a first-token
latency plot alongside the streaming-latency example above:

.. code-block:: toml

   [[reports.pareto.plots]]
   name = "First-token latency vs throughput"
   x = { metric = "ttft", match = { statistic = "p99" } }
   y = { metric = "output_token_throughput" }
   frontiers = [{ name = "All configurations" }]

Dynamo Phases
~~~~~~~~~~~~~

Each AIPerf phase with recorded metrics contributes separately. For example, phases with concurrency ``2``
and ``4`` produce two independent points per successful iteration or sweep step. Point details show all X and Y
metric dimensions alongside the phase's benchmark settings. Phases without observations contribute no points.
The final-phase copy in ``aiperf_report.csv`` is not an extra point.

To compare phases separately, select their configured names with ``match.phase``:

.. code-block:: toml

   [[reports.pareto.plots]]
   name = "Throughput by phase"
   x = { metric = "ttft", match = { statistic = "p99" } }
   y = { metric = "output_token_throughput" }
   frontiers = [
     { name = "Round 1", match = { phase = "round_1" } },
     { name = "Round 2", match = { phase = "round_2" } },
   ]

When an execution runs both AIPerf and GenAI-Perf, their points also have a ``benchmark`` dimension
(``aiperf`` or ``genai_perf``). Use ``match.benchmark`` to select one. Optional accuracy results describe the
separate accuracy evaluation and are shared across that execution's phases.

Comparing Versions or Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Give each version its own frontier so its measurements compete independently. Append this plot to an enabled
inference report, replacing the ``cases`` entries with your scenario's ``[[Tests]].id`` values:

.. code-block:: toml

   [[reports.pareto.plots]]
   name = "Streaming efficiency by version"
   x = { metric = "tpot", match = { statistic = "p99" } }
   y = { metric = "output_token_throughput_per_serving_gpu" }
   frontiers = [
     { name = "Version A", cases = ["a-c16", "a-c32", "a-c64"] },
     { name = "Version B", cases = ["b-grid"] },
   ]

Here, Version A combines three cases into one frontier; Version B uses every successful point from one sweep
case. Combining both versions into a single group would instead select the best points across both versions.

``direction = "minimize"`` or ``direction = "maximize"`` overrides which values are preferred on an axis.
Both axes retain ascending numeric scales. Use the default directions listed above for these inference metrics.

Viewing and Regenerating Reports
--------------------------------

Charts initially show frontier points. Enable **Show dominated points** to inspect the other measurements.
Click a point to pin its details for selection and copying, including inference settings and the executed TOML.
The measurements table includes every valid execution, including ties. Lines connect measured points; they do
not represent additional measurements between those points.

To change plots without rerunning workloads, edit the scenario and regenerate reports from saved results:

.. code-block:: bash

   cloudai generate-report --system-config system.toml --tests-dir tests/ \
     --test-scenario scenario.toml --result-dir results/my-scenario
