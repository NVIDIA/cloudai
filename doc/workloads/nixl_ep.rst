NIXL EP
=======

This workload (``test_template_name`` is ``NixlEP``) runs the NIXL Elastic EP benchmark through a Slurm-managed multi-node elastic launcher flow.

Overview
--------

The Slurm launch model is:

- one ``elastic.py`` process per node, started in sequence as the plan progresses
- the master node starts first and exposes a TCPStore for rank coordination
- follower nodes connect via ``--tcp-server $master_ip`` once the master is ready
- the benchmark runtime comes from the container image
- each run serializes its plan JSON into the output directory

Plan Format
-----------

The ``plan`` field is a JSON-encoded list of phases. Each phase is a list of rank indices passed directly to the benchmark. CloudAI uses the following convention to drive the elastic launcher:

- **Positive rank index** — the rank is active. A rank that is new relative to the previous phase causes CloudAI to fire an additional ``srun`` for that worker.
- **Negative rank index** (e.g. ``-6``) — signals a contraction: the benchmark sees the absolute value and treats it as temporarily removed. No new ``srun`` is launched for negative indices.
- **Omitted rank** — a rank present in an earlier phase but absent from the current phase list is not relaunched. The benchmark's own phase logic handles its inactivity.

Example:

.. code-block:: text

   [[0, 1, 2, 3],              # phase 0: ranks 0–3 start
    [0, 1, 2, 3, 4, 5, 6, 7], # phase 1: ranks 4–7 join (expansion)
    [0, 1, 2, 3, 4, -6, 7],   # phase 2: rank 6 contracted (no new launch)
    [0, 1, 2, 3, 4, 5, 6, 7]] # phase 3: rank 6 rejoins (new launch for rank 6)

Phase completion is detected by polling the primary log for ``-> end phase N`` markers.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "nixl-ep-expansion-contraction"
   description = "NIXL Elastic EP expansion/contraction benchmark"
   test_template_name = "NixlEP"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   elastic_script = "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py"
   plan = "[[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, -6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]"
   num_processes_per_node = 4
   num_tokens = 256
   num_experts_per_rank = 4
   hidden_dim = 8192
   num_topk = 6
   disable_ll_nvlink = true

Test-in-Scenario example:

.. code-block:: toml

   name = "nixl-ep-expansion-contraction"

   [[Tests]]
   id = "nixl_ep.expansion_contraction"
   num_nodes = 3
   time_limit = "00:30:00"

   name = "nixl-ep-expansion-contraction"
   description = "NIXL Elastic EP expansion/contraction benchmark"
   test_template_name = "NixlEP"

     [Tests.cmd_args]
     docker_image_url = "<docker container url here>"
     elastic_script = "/workspace/nixl/examples/device/ep/tests/elastic/elastic.py"
     plan = "[[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, -6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]"
     num_processes_per_node = 4
     num_tokens = 256
     num_experts_per_rank = 4
     hidden_dim = 8192
     num_topk = 6
     disable_ll_nvlink = true

Reporting
---------

After a run completes, CloudAI prints a single table with one row per (node, rank) measurement. The ``Phases`` column shows each phase index colour-coded green (passed) or red (failed). Bandwidth columns report dispatch+combine throughput and timing per rank.

The reported metric (``default``) is the mean dispatch+combine bandwidth in GB/s across all ranks.

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autopydantic_model:: cloudai.workloads.nixl_ep.nixl_ep.NixlEPCmdArgs
   :members:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nixl_ep.nixl_ep.NixlEPTestDefinition
   :members:
   :show-inheritance:
