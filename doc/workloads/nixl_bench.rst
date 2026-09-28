NIXL Bench
==========

This workload (`test_template_name` is ``NIXLBench``) runs NIXL benchmarking suite for network and interconnect performance testing.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_nixl_bench_test"
   description = "Example NIXL Bench test"
   test_template_name = "NIXLBench"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   path_to_benchmark = "/workspace/nixlbench/build/nixlbench"
   backend = "UCX"
   initiator_seg_type = "VRAM"
   target_seg_type = "VRAM"
   op_type = "READ"
   filepath = "/data"
   device_list = "11:F:/store0.bin"
   # one could also use <num>kb, <num>mb, <num>gb shortcuts
   total_buffer_size = 8000000000

Test Scenario example:

.. code-block:: toml

   name = "nixl-bench-test"

   [[Tests]]
   id = "bench.1"
   num_nodes = 1
   time_limit = "00:10:00"

   test_name = "my_nixl_bench_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "nixl-bench-test"

   [[Tests]]
   id = "bench.1"
   num_nodes = 1
   time_limit = "00:10:00"

   name = "my_nixl_bench_test"
   description = "Example NIXL Bench test"
   test_template_name = "NIXLBench"

     [Tests.cmd_args]
     docker_image_url = "<docker container url here>"
     path_to_benchmark = "/workspace/nixlbench/build/nixlbench"
     backend = "UCX"
     initiator_seg_type = "DRAM"
     target_seg_type = "DRAM"
     op_type = "WRITE"

Runtime Coordination
--------------------

NIXLBench uses ETCD by default. CloudAI starts ETCD from the benchmark image when
``etcd_image_url`` is omitted, or from the configured image.

To use NIXLBench's direct two-process ASIO runtime instead, set:

.. code-block:: toml

   runtime_type = "ASIO"

CloudAI resolves ``asio_address`` to the first allocated node by default and does not
install or start ETCD. ``asio_address`` and ``asio_port`` can be overridden explicitly.
ASIO requires exactly two NIXLBench processes. For UCX, a one-node test runs both
processes locally, while a two-node test runs one process on each node.

Storage backends can run without either runtime by using an empty ETCD endpoint:

.. code-block:: toml

   backend = "POSIX"
   etcd_endpoints = ""

This null-runtime mode is limited to storage backends and launches one NIXLBench process by default.

Independent Storage Processes
-----------------------------

To run independent storage benchmarks in one allocation, set
``launch_mode = "independent"`` and ``etcd_endpoints = ""``. CloudAI launches one
process per node by default; use ``num_nodes`` to choose the number of nodes:

.. code-block:: toml

   name = "storage-scaling"

   [[Tests]]
   id = "storage.4nodes"
   test_name = "my_storage_test"
   num_nodes = 4

     [Tests.cmd_args]
     launch_mode = "independent"
     etcd_endpoints = ""
     num_threads = [4, 8]

Here, ``my_storage_test`` is a NIXLBench test defining the storage backend, container
image, benchmark path and workload arguments. CloudAI runs the 4-thread and 8-thread
configurations as separate tests.

``launch_mode`` selects CloudAI's launch strategy and is not passed to NIXLBench.
Its default value, ``"default"``, preserves existing behavior, including when task
placement options are present. Placement options alone do not enable independent runs.

Independent mode requires ``etcd_endpoints = ""`` and the default
``runtime_type = "ETCD"`` setting, which together select NIXLBench's null runtime.
Managed/external ETCD and ASIO cannot be used with this mode. CloudAI launches one
Slurm step across the requested nodes. Use ``extra_srun_args`` only when overriding
the default task placement. System-level options precede test-level options.

Each task executes in its own shell. Node-specific environment values and command
expansions are evaluated there. CloudAI does not discover backend devices.

Results and Reporting
~~~~~~~~~~~~~~~~~~~~~

Independent runs retain ``nixlbench/<task-id>.stdout``, ``.stderr``, ``.status`` and
``.hostname`` files. The ``nixlbench/ntasks`` file records the actual Slurm task count.
A failed process, missing task output or inconsistent block-size/batch-size coverage
makes the run unsuccessful; incomplete results are not aggregated.

``nixlbench_per_task.csv`` preserves individual measurements with task IDs and hostnames.
``nixlbench.csv`` summarizes matching block sizes and batch sizes: ``avg_lat`` and
``bw_gb_sec`` are arithmetic means across tasks, ``bw_min_gb_sec`` is the minimum
task bandwidth, ``bw_sum_gb_sec`` is the sum, and ``task_count`` records the contributors.
The native HTML report shows mean latency and minimum/mean/summed bandwidth, with
separate plots for each batch size. Comparison reports and metric observations use
the mean per-task bandwidth and latency. Existing single-process reports are unchanged.

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchCmdArgs
   :members:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_bench.nixl_bench.NIXLBenchTestDefinition
   :members:
   :show-inheritance:
