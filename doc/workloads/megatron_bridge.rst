MegatronBridge
==============

This workload (`test_template_name` is ``MegatronBridge``) submits training and finetuning tasks based on Megatron-Bridge framework.

.. note::

   This workload has a hard requirement for the HuggingFace Hub token. There are two options:

   - (recommended) define ``HF_TOKEN`` environment variable
   - set ``cmd_args.hf_token`` either in Test or Scenario config


Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "megatron_bridge_qwen_30b"
   description = "Megatron-Bridge run via CloudAI SlurmSystem for Qwen3 30B A3B"
   test_template_name = "MegatronBridge"

   [[git_repos]]
   url = "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
   commit = "v0.3.0"
   mount_as = "/opt/Megatron-Bridge"

   [cmd_args]
   gpu_type = "gb200"
   gpus_per_node = 8
   num_gpus = 8
   # Container can be an NGC/enroot URL (nvcr.io#...) or a local .sqsh path.
   container_image = "nvcr.io#nvidia/nemo:26.02.00"

   model_family_name = "qwen"
   model_recipe_name = "qwen3_30b_a3b"
   task = "pretrain"
   domain = "llm"
   compute_dtype = "fp8_mx"

Test Scenario example:

.. code-block:: toml

   name = "megatron_bridge_qwen_30b"

   [[Tests]]
   id = "megatron_bridge_qwen_30b"
   test_name = "megatron_bridge_qwen_30b"
   num_nodes = "2"

Test-in-Scenario example:

.. code-block:: toml

   name = "megatron-bridge-test"

   [[Tests]]
   id = "mbridge.1"
   num_nodes = 2
   time_limit = "00:30:00"

   name = "megatron_bridge_qwen_30b"
   description = "Megatron-Bridge run via CloudAI SlurmSystem for Qwen3 30B A3B"
   test_template_name = "MegatronBridge"

     [[Tests.git_repos]]
     url = "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
     commit = "v0.3.0"
     mount_as = "/opt/Megatron-Bridge"

     [Tests.cmd_args]
     container_image = "nvcr.io#nvidia/nemo:26.02.01"
     model_family_name = "qwen"
     model_recipe_name = "qwen3_30b_a3b"

     gpu_type = "gb200"
     gpus_per_node = 8
     num_gpus = 8

     task = "pretrain"
     domain = "llm"
     compute_dtype = "fp8_mx"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.megatron_bridge.megatron_bridge.MegatronBridgeCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.megatron_bridge.megatron_bridge.MegatronBridgeTestDefinition
   :members:
   :show-inheritance:
