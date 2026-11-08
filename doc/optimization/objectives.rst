Choose an objective and search space
====================================

Define the decision before selecting the search dimensions. Optimize a measured quantity whose units and direction you understand.

Start with one metric
---------------------

Choose a metric exposed by the workload's reporter. The search ranks valid rewards in descending order.
Use the identity reward for a quantity to maximize, or a supported inverse/negative reward for a quantity to minimize,
checking the transform's domain and failure behavior before running.
The two-choice :doc:`search` recipe uses the NCCL reporter's ``default`` metric and inverse reward.

Define meaningful choices
-------------------------

List-valued command arguments, environment variables, and node counts can define search dimensions.
Keep model, input distribution, and measurement conditions fixed unless they are intentional dimensions.
If an argument is inherently a list rather than a set of alternatives, use ``dse_excluded_args`` as described in
:doc:`/reference/agents`.

Interpret multiple metrics
--------------------------

Inspect latency, throughput, and resource use together when one number hides a trade-off.
:doc:`/pareto-frontiers` explains comparison plots. A Pareto report describes observed trade-offs; the selected
configuration still depends on the configured reward. Missing or failed metrics have explicit handling in
:doc:`/reference/agents` and must not be interpreted as successful measurements.
