Agent and search settings
=========================

Built-in ``grid_search`` evaluates every configured combination; ``agent_steps`` is not a grid-search cap.
See :doc:`/optimization/search` for an executable example.

For DSE workloads, you can pass agent configuration via ``agent_config`` in the scenario.

``BaseAgentConfig`` includes:

- ``random_seed``: controls deterministic random behavior in agents.
- ``start_action``: controls the very first action strategy (``"random"`` or ``"first"``).
- ``rewards`` (optional): nested table mapped to ``RewardOverrides``. When present, ``CloudAIGymEnv`` uses it for
  constraint failures and for substituting failed-metric values in observations.

  - ``constraint_failure``: reward returned when ``TestDefinition.constraint_check`` fails on a step. If omitted
    or unset, the environment uses ``-1.0``.
  - ``metric_failure``: value written into the observation vector when a metric is missing or is the
    canonical error sentinel ``METRIC_ERROR`` (a distinct object from numeric metrics, so a valid ``-1.0``
    result is not mistaken for an error). If omitted or unset, observations use ``-1.0`` for that slot.

Example:

.. code-block:: toml

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   agent = "grid_search"
   agent_metrics = ["default"]
   agent_reward_function = "inverse"

     [Tests.agent_config]
     random_seed = 123
     start_action = "first"

       [Tests.agent_config.rewards]
       constraint_failure = -5.0
       metric_failure = 0.0

When an agent honors ``start_action = "first"``, it should start from ``CloudAIGymEnv.first_sweep`` (the sweep
is built from first values of each sweep parameter). ``start_action = "random"`` means starting from a random
action, typically seeded by ``random_seed``.

Custom agents may extend the ``BaseAgentConfig`` and offer more parameters to configure.

DSE Parameter Exclusions
~~~~~~~~~~~~~~~~~~~~~~~~

CloudAI builds the DSE parameter space implicitly from list-valued fields under ``cmd_args``, list-valued
``extra_env_vars``, and list-valued ``num_nodes``. Most lists mean "try each value", but some workload settings are
real list-valued configuration, such as worker port lists or ordered benchmark phases.

Use ``dse_excluded_args`` when a list under ``cmd_args`` should stay intact instead of becoming a sweep dimension.
Entries must be dot-separated paths that start with ``cmd_args.`` and may point to either a single field or a parent
object. Matching is prefix-based, so excluding ``cmd_args.foo`` also excludes nested list-valued fields such as
``cmd_args.foo.bar`` from DSE parameter discovery.

.. code-block:: toml

   [[Tests]]
   id = "Tests.1"
   test_name = "my_test"
   dse_excluded_args = ["cmd_args.lmcache.lmcache_worker_ports"]

     [Tests.cmd_args.lmcache]
     chunk_size = [256, 512]
     lmcache_worker_ports = [8788, 8789, 8790, 8791]

In this example, ``cmd_args.lmcache.chunk_size`` is still swept, while
``cmd_args.lmcache.lmcache_worker_ports`` is passed through as one list value. The exclusion does not remove or mutate
the field; it only prevents CloudAI from adding that path to the DSE parameter space.

``dse_excluded_args`` currently applies only to ``cmd_args`` paths. It does not exclude list-valued ``extra_env_vars``
or ``num_nodes``; those lists are still interpreted as sweep dimensions. To exclude many nested list fields at once,
exclude their common parent path. Common examples are ``cmd_args.aiperf_phases`` and
``cmd_args.lmcache.lmcache_worker_ports``.

Metric Errors and Report Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Report generation strategies signal a failed or missing metric by returning the singleton ``METRIC_ERROR`` from
``cloudai.core`` (type ``MetricErrorSentinel``; ``float(METRIC_ERROR)`` is ``-1.0`` for numeric use). The declared
return type of ``ReportGenerationStrategy.get_metric`` is ``MetricValue`` (``float | MetricErrorSentinel``).

When branching on success vs failure, use **identity** (``value is METRIC_ERROR``), not equality to ``-1.0``, so a legitimate metric of ``-1.0`` is not
treated as an error. The gym observation path uses that identity check before applying ``agent_config.rewards.metric_failure``.
