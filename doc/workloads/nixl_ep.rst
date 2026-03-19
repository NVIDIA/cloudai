NIXL Elastic EP
===============

This workload (``test_template_name`` is ``NixlEP``) runs the NIXL Elastic EP benchmark through a Slurm-managed multi-node launcher flow.

Overview
--------

The Slurm launch model is:

- one ``elastic.py`` launcher per node
- the master node starts first
- follower nodes connect with ``--tcp-server $master_ip``
- the benchmark runtime comes from the container image
- each test case serializes its own plan JSON into the run output directory

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
   num_processes_per_node = [4, 4, 2]
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
     num_processes_per_node = [4, 4, 2]
     num_tokens = 256
     num_experts_per_rank = 4
     hidden_dim = 8192
     num_topk = 6
     disable_ll_nvlink = true

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
