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
- optional ``[[git_repos]]`` mounts are for custom input JSON files only

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
   input_json = "/workspace/nixl/examples/device/ep/tests/elastic/expansion_contraction.json"
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
     input_json = "/workspace/nixl/examples/device/ep/tests/elastic/expansion_contraction.json"
     num_processes_per_node = [4, 4, 2]
     num_tokens = 256
     num_experts_per_rank = 4
     hidden_dim = 8192
     num_topk = 6
     disable_ll_nvlink = true

Optional config repo example:

.. code-block:: toml

   [[git_repos]]
   url = "https://github.com/NVIDIA/nixl-configs.git"
   commit = "main"
   mount_as = "/workspace/nixl-ep-configs"

   [cmd_args]
   input_json = "/workspace/nixl-ep-configs/plans/custom_plan.json"

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
