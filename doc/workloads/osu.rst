OSU
===

This workload (``test_template_name`` is ``OSUBench``) allows you to execute OSU Micro Benchmarks
within the CloudAI framework.

Usage example
-------------

Test example:

.. code-block:: toml

    name = "osu_example"
    test_template_name = "OSUBench"
    description = "OSU Benchmark example"

    [cmd_args]
    "docker_image_url" = "docker-image-with-osu-benchmark:latest"
    "benchmarks_dir" = "/directory/with/osu/binaries/in/container"
    "benchmark" = ["osu_allreduce", "osu_allgather"]
    "iterations" = 10
    "message_size" = "1024"

Test Scenario example:

.. code-block:: toml

    name = "osu_example"

    [[Tests]]
    id = "Tests.1"
    test_name = "osu_example"
    num_nodes = "2"
    time_limit = "00:20:00"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.osu_bench.osu_bench.OSUBenchCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.osu_bench.osu_bench.OSUBenchTestDefinition
   :members:
   :show-inheritance:
