MegatronBridge
==============

This workload (`test_template_name` is ``MegatronBridge``) submits training and finetuning tasks based on Megatron-Bridge framework.


Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "megatron_bridge_qwen_30b"
   description = "Megatron-Bridge run via CloudAI SlurmSystem for Qwen3 30B A3B"
   test_template_name = "MegatronBridge"

   [cmd_args]
   # Container can be an NGC/enroot URL (nvcr.io#...) or a local .sqsh path.
   container_image = "nvcr.io#nvidia/nemo:25.11.01"

   model_name = "qwen3"
   model_size = "30b_a3b"
   task = "pretrain"
   domain = "llm"
   compute_dtype = "fp8_mx"

   hf_token = "hf_xxx"

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

     [Tests.cmd_args]
     container_image = "nvcr.io#nvidia/nemo:25.11.01"
     model_name = "qwen3"
     model_size = "30b_a3b"
     task = "pretrain"
     domain = "llm"
     compute_dtype = "fp8_mx"
     hf_token = "hf_xxx"


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
