Choose an objective and search space
====================================

Define the decision before selecting the search dimensions. Optimize a measured quantity whose units and direction you understand.

Start with one metric
---------------------

Choose a metric exposed by the workload's reporter. Search ranks valid rewards in descending order.

.. list-table::
   :header-rows: 1

   * - Goal
     - Reward
     - Check
   * - Maximize a metric
     - Identity
     - Higher values represent the desired outcome.
   * - Minimize a metric
     - Inverse or negative
     - The transform accepts the metric domain and handles failures correctly.

For the NCCL search, add these settings to the case:

.. code-block:: toml

   agent_metrics = ["default"]
   agent_reward_function = "inverse"

The NCCL ``default`` metric is mean in-place latency in microseconds across reported message sizes.
Use the complete :doc:`search` example to run it.


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
