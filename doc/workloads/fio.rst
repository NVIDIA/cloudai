fio
===

``Fio`` runs the open-source ``fio`` tool from CloudAI.
It supports standalone and Slurm systems.

Scenarios
---------

Use ``conf/common/test_scenario/fio.toml`` for Slurm clusters.
It runs the common ``fio`` test in a container and starts 4 tasks total:
``num_nodes = 2`` and ``num_tasks_per_node = 2``.

Use ``conf/common/test_scenario/fio_local.toml`` for local/standalone runs.
It uses host-installed ``fio`` and no container image.

Fio Arguments
-------------

Put fio CLI options under ``cmd_args.args``.
CloudAI does not declare a fixed fio option list; keys pass through verbatim.

.. code-block:: toml

    name = "fio"
    description = "fio random write smoke test"
    test_template_name = "Fio"

    [cmd_args]
    fio_binary = "fio"

      [cmd_args.args]
      name = "fio-smoke"
      filename = "/tmp/cloudai-fio-test"
      rw = "randwrite"
      bs = "128k"
      size = "80m"
      iodepth = 1
      numjobs = 1
      group_reporting = true
      thread = true

This emits options like ``--rw=randwrite`` and ``--group_reporting``.
Quote TOML keys when option names contain characters such as ``-``:

.. code-block:: toml

      [cmd_args.args]
      "max-jobs" = 4

List values are CloudAI DSE sweeps:

.. code-block:: toml

      [cmd_args.args]
      bs = ["128k", "1m"]
      iodepth = [1, 8, 32]

To repeat the same fio option in one run, use a nested table.
Nested key names are only stable TOML item names; values become fio values:

.. code-block:: toml

      [cmd_args.args.a]
      "0" = "=foo"  # --a==foo
      "1" = "bar"   # --a=bar

      [cmd_args.args."client"]
      "0" = "host1" # --client=host1
      "1" = "host2" # --client=host2

For existing or complex fio configs, use ``job_file``:

.. code-block:: toml

    [cmd_args]
    fio_binary = "/tmp/fio/fio"
    job_file = "/tmp/kv_emulation.fio"

Slurm Tasks
-----------

Scenario ``num_nodes`` controls node count.
``cmd_args.num_tasks_per_node`` controls Slurm tasks per node.
Total fio tasks = ``num_nodes * num_tasks_per_node``.
CloudAI passes this as ``--ntasks`` and ``--ntasks-per-node``.

.. code-block:: toml

    [[Tests]]
    id = "Tests.fio"
    test_name = "fio"
    num_nodes = 2

      [Tests.cmd_args]
      docker_image_url = "openeuler/fio:3.42-oe2403sp3"
      num_tasks_per_node = 2

Default Metric
--------------

For ``agent_metrics = ["default"]``, CloudAI aggregates parsed fio summary rows.
Defaults report total bandwidth across all operations and tasks.
Bandwidth metrics are normalized to MiB/s before aggregation.
Latency metrics are normalized to usec before aggregation.

.. code-block:: toml

    [cmd_args]
    metric_operation = "all"    # read, write, trim, all, first
    metric_name = "bw"          # bw, iops, latency
    metric_aggregate = "sum"    # sum, mean, min, max, first

Raw parsed rows and normalized metric values are written to ``fio_summary.csv``.

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.fio.fio.FioCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.fio.fio.FioTestDefinition
   :members:
   :show-inheritance:
