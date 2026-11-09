Workload and system support
===========================

Use this matrix to choose an execution system for your workload. Names match ``test_template_name`` in TOML.
A check mark means this revision registers an execution strategy for that combination; workload-specific
prerequisites still apply.

.. _workload-system-support:

Compatibility
-------------

.. csv-table::
   :header: "Workload", "Slurm", "Standalone", "Kubernetes", "RunAI", "LSF"
   :widths: 35, 13, 13, 13, 13, 13

   ":doc:`Aiconfigurator </workloads/aiconfigurator>`", "—", "✅", "—", "—", "—"
   ":doc:`AIDynamo </workloads/ai_dynamo>`", "✅", "—", "✅", "—", "—"
   ":doc:`BashCmd </workloads/bash_cmd>`", "✅", "—", "—", "—", "—"
   ":doc:`ChakraReplay </workloads/chakra_replay>`", "✅", "—", "—", "—", "—"
   ":doc:`DDLBTest </workloads/ddlb>`", "✅", "—", "—", "—", "—"
   ":doc:`DeepEP </workloads/deepep>`", "✅", "—", "—", "—", "—"
   ":doc:`DynamoMocker </workloads/dynamo_mocker>`", "—", "✅", "—", "—", "—"
   ":doc:`Fio </workloads/fio>`", "✅", "✅", "—", "—", "—"
   ":doc:`JaxToolboxGPT </workloads/jax_toolbox>`", "✅", "—", "—", "—", "—"
   ":doc:`JaxToolboxGrok </workloads/jax_toolbox>`", "✅", "—", "—", "—", "—"
   ":doc:`JaxToolboxNemotron </workloads/jax_toolbox>`", "✅", "—", "—", "—", "—"
   ":doc:`MegatronBridge </workloads/megatron_bridge>`", "✅", "—", "—", "—", "—"
   "``MegatronRun``", "✅", "—", "—", "—", "—"
   "``MoEBenchmark``", "✅", "—", "—", "—", "—"
   ":doc:`NcclTest </workloads/nccl>`", "✅", "—", "✅", "✅", "—"
   ":doc:`NeMoLauncher </workloads/nemo_launcher>`", "✅", "—", "—", "—", "—"
   ":doc:`NeMoRun </workloads/nemo_run>`", "✅", "—", "—", "—", "—"
   ":doc:`NIXLBench </workloads/nixl_bench>`", "✅", "—", "—", "—", "—"
   ":doc:`NixlEP </workloads/nixl_ep>`", "✅", "—", "—", "—", "—"
   ":doc:`NIXLKVBench </workloads/nixl_kvbench>`", "✅", "—", "—", "—", "—"
   ":doc:`NixlPerftest </workloads/nixl_perftest>`", "✅", "—", "—", "—", "—"
   ":doc:`OSUBench </workloads/osu>`", "✅", "—", "—", "—", "—"
   ":doc:`sglang </workloads/sglang>`", "✅", "—", "—", "—", "—"
   ":doc:`Sleep </workloads/sleep>`", "✅", "✅", "✅", "—", "✅"
   ":doc:`SlurmContainer </workloads/slurm_container>`", "✅", "—", "—", "—", "—"
   ":doc:`SRTSlurm </workloads/srt_slurm>`", "✅", "—", "—", "—", "—"
   "``TritonInference``", "✅", "—", "—", "—", "—"
   ":doc:`UCCTest </workloads/ucc>`", "✅", "—", "—", "—", "—"
   ":doc:`vllm </workloads/vllm>`", "✅", "—", "—", "—", "—"

Choose a setup path
-------------------

* :doc:`/guides/slurm`: submit workloads to compute nodes from a login or submit host.
* :doc:`/Tutorial`: run a local storage experiment with Standalone.
* :doc:`/guides/kubernetes`: run Dynamo or NCCL with the appropriate Kubernetes operators.

RunAI currently has an NCCL integration; use :doc:`/guides/runai` for its configuration.
LSF currently has a Sleep integration for scheduling checks; its fields are under :ref:`lsf-system`.
