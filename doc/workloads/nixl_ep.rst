NIXL Elastic EP
===============

This workload (``test_template_name`` is ``NixlEP``) runs the NIXL Elastic EP benchmark through a Slurm-managed multi-node launcher flow.

Overview
--------

The Slurm launch model is:

- one ``elastic.py`` launcher per node
- the master node starts first
- follower nodes connect with ``--tcp-server $master_ip``
- the benchmark repo is provided via ``[[git_repos]]`` and mounted inside the container

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "nixl-ep-expansion-contraction"
   description = "NIXL Elastic EP expansion/contraction benchmark"
   test_template_name = "NixlEP"

   [cmd_args]
   docker_image_url = "<docker container url here>"
   elastic_script = "tests/elastic/elastic.py"
   input_json = "tests/elastic/expansion_contraction.json"
   num_processes_per_node = [4, 4, 2]
   num_tokens = 256
   num_experts_per_rank = 4
   hidden_dim = 8192
   num_topk = 6
   disable_ll_nvlink = true

   [extra_env_vars]
   NIXL_PLUGIN_DIR = "/workspace/nixl/lib/x86_64-linux-gnu/plugins"
   LD_LIBRARY_PATH = "/workspace/rdma_core/lib:$LD_LIBRARY_PATH"
   PYTHONPATH = "/workspace/nixl/examples/device/ep"

   [[git_repos]]
   url = "https://github.com/NVIDIA/nixl.git"
   commit = "main"
   mount_as = "/workspace/nixl"

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
     elastic_script = "tests/elastic/elastic.py"
     input_json = "tests/elastic/expansion_contraction.json"
     num_processes_per_node = [4, 4, 2]
     num_tokens = 256
     num_experts_per_rank = 4
     hidden_dim = 8192
     num_topk = 6
     disable_ll_nvlink = true

     [Tests.extra_env_vars]
     NIXL_PLUGIN_DIR = "/workspace/nixl/lib/x86_64-linux-gnu/plugins"
     LD_LIBRARY_PATH = "/workspace/rdma_core/lib:$LD_LIBRARY_PATH"
     PYTHONPATH = "/workspace/nixl/examples/device/ep"

     [[Tests.git_repos]]
     url = "https://github.com/NVIDIA/nixl.git"
     commit = "main"
     mount_as = "/workspace/nixl"

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
