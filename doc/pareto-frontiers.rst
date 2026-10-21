Pareto Frontiers
================

The ``pareto`` scenario report writes ``pareto.html`` in the scenario results directory. Each plot compares two
metrics. Successful iterations and sweep steps supply the points: one test case can contribute several points,
and several cases can contribute to one frontier.

Usage Example
-------------

For :doc:`vLLM <workloads/vllm>` or :doc:`SGLang <workloads/sglang>`, add this block to your scenario TOML:

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
from each execution. ``match`` selects among measurements of the same metric, such as different statistics or
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
