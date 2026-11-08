Prepare model weights
=====================

Choose the model in the workload configuration before allocating compute. Use a small public model to verify the execution path.

Start small
-----------

The :doc:`/workloads/vllm` starter uses ``Qwen/Qwen3-0.6B``. Run it through the
:ref:`inference example <inference-with-vllm>` after preparing Slurm. Ensure the workload can reach its model source.

Use existing weights
--------------------

For vLLM, point the model argument at a container-visible directory. In the inline scenario's test entry, use:

.. code-block:: toml

   extra_container_mounts = ["/shared/models:/models"]

   [Tests.cmd_args]
   model = "/models/my-model"

Replace both paths with your model location and retain the starter's other required settings.
For training, use the selected workload's dataset and tokenizer settings.


Access and storage
------------------

Before submission:

* Authenticate when the model provider requires it. Pass credentials through the workload's environment.
* Set ``HF_TOKEN`` for the Megatron Bridge starter; keep tokens out of scenario files.
* Allow space for the selected revision and verify its download against provider metadata.


For larger models, first confirm device memory, parallelism, and shared-storage access on the intended nodes.
Keep the model revision and traffic or training inputs fixed when comparing configurations.
