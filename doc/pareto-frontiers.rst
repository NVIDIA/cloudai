Pareto frontiers
================

Enable ``pareto`` in a scenario to generate ``pareto.html``. Each plot compares two metrics and excludes a measurement
when another is at least as good on both axes and better on at least one. All successful iterations and DSE steps
participate: a grid-search case can contribute several points, and several cases can contribute to one frontier.

Choose the axes
---------------

For inference, use latency on X and throughput on Y. vLLM and SGLang provide:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - X: lower is better
     - Y: higher is better
     - Comparison
   * - ``tpot`` (ms)
     - ``output_token_throughput_per_serving_gpu`` (tokens/s/GPU)
     - Streaming responsiveness versus serving efficiency
   * - ``ttft`` (ms)
     - ``output_token_throughput`` (tokens/s)
     - First-token wait versus total capacity

These are throughput/latency tradeoffs used in inference benchmarking. Publications such as
`SemiAnalysis InferenceX <https://inferencex.semianalysis.com/blog/inferencex-v2-nvidia-blackwell-vs-amd-vs-hopper>`_
also show streaming interactivity on X: the reciprocal of TPOT, e.g. 20 ms/token corresponds to 50 tokens/s/user.
Using TPOT directly selects the same frontier when the statistic and throughput definition match, with different
X coordinates. Aggregate throughput divided by configured concurrency is not that streaming-speed metric.

Select ``mean``, ``median``, or ``p99`` with ``match.statistic`` for TTFT and TPOT. Match the publication's statistic,
token counts, and GPU denominator when comparing results. TTFT measures the initial wait; TPOT measures generation
after the first token. Keep model, input/output lengths, and traffic conditions comparable within a frontier.

One frontier, all cases
-----------------------

Add this to the scenario TOML:

.. code-block:: toml

   [reports.pareto]
   enable = true

   [[reports.pareto.plots]]
   name = "Streaming latency vs serving efficiency"
   x = { metric = "tpot", match = { statistic = "p99" } }
   y = { metric = "output_token_throughput_per_serving_gpu" }
   frontiers = [{ name = "All configurations" }]

Omitting ``cases`` or setting ``cases = []`` includes every case in the scenario. Optimization follows the metric:
latency is minimized and throughput maximized. Both axes increase normally; a latency/throughput frontier rises
as higher throughput requires accepting higher latency.

Several plots and comparison groups
-----------------------------------

Append another ``[[reports.pareto.plots]]`` for a second chart. Give each version its own frontier to compare their
best configurations independently:

.. code-block:: toml

   [[reports.pareto.plots]]
   name = "First-token latency by version"
   x = { metric = "ttft", match = { statistic = "p99" } }
   y = { metric = "output_token_throughput_per_allocated_gpu" }

   [[reports.pareto.plots.frontiers]]
   name = "Version A"
   cases = ["a-c16", "a-c32", "a-c64"]

   [[reports.pareto.plots.frontiers]]
   name = "Version B"
   cases = ["b-grid"]

``cases`` contains scenario ``[[Tests]].id`` values. A single search case and multiple fixed-concurrency cases
are handled identically: all their measurements compete within the named frontier.

For Y, ``output_token_throughput_per_allocated_gpu`` divides output tokens/s by the Slurm reservation, including
idle GPUs; in single-sbatch mode this is the whole batch's reservation. ``output_token_throughput_per_serving_gpu``
uses the configured serving layout, summing prefill and decode GPUs. It describes configured devices, not measured
GPU utilization. Counts appear in point details; unavailable counts leave the corresponding metric unavailable.

Reading and regenerating the report
-----------------------------------

Charts initially show frontier points. Enable **Show dominated points** to inspect the other measurements.
Click a point to pin its details for selection and copying, including inference settings and the executed TOML.
The measurements table retains every valid execution, including ties. Failed executions and missing or ambiguous
metrics appear as exclusions. Lines connect measured points; intermediate configurations have not been measured.

To change plots without rerunning workloads, edit the scenario and regenerate reports from saved results:

.. code-block:: bash

   cloudai generate-report --system-config system.toml --tests-dir tests/ \
     --test-scenario scenario.toml --result-dir results/my-scenario

Other workloads can use any pair of metrics they expose through ``metric_observations()``. Use ``match`` on an axis
or frontier to select measurement dimensions; each axis must select exactly one observation per execution.
Shared dimensions must agree between axes, except for the statistic. Override an axis's optimization preference
with ``direction = "minimize"`` or ``"maximize"`` when needed.
