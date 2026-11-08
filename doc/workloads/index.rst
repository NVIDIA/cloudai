Workloads
=========

Choose a workload family, then check compatibility with your execution system.

.. toctree::
   :maxdepth: 1

   training
   inference
   networking
   storage
   simulation
   general
   legacy

Compatibility
-------------

.. csv-table::
   :header: "Test", "Slurm", "Kubernetes", "RunAI", "Standalone"
   :widths: 40, 15, 15, 15, 15

   ":doc:`aiconfigurator`", "❌", "❌", "❌", "✅"
   ":doc:`ai_dynamo`", "✅", "✅", "❌", "❌"
   ":doc:`bash_cmd`", "✅", "❌", "❌", "❌"
   ":doc:`chakra_replay`", "✅", "❌", "❌", "❌"
   ":doc:`ddlb`", "✅", "❌", "❌", "❌"
   ":doc:`deepep`", "✅", "❌", "❌", "❌"
   ":doc:`dynamo_mocker`", "❌", "❌", "❌", "✅"
   ":doc:`fio`", "✅", "❌", "❌", "✅"
   ":doc:`jax_toolbox`", "✅", "❌", "❌", "❌"
   "MegatronRun", "✅", "❌", "❌", "❌"
   ":doc:`megatron_bridge`", "✅", "❌", "❌", "❌"
   ":doc:`nccl`", "✅", "✅", "✅", "❌"
   ":doc:`nemo_launcher`", "✅", "❌", "❌", "❌"
   ":doc:`nemo_run`", "✅", "❌", "❌", "❌"
   ":doc:`nixl_bench`", "✅", "❌", "❌", "❌"
   ":doc:`nixl_ep`", "✅", "❌", "❌", "❌"
   ":doc:`nixl_kvbench`", "✅", "❌", "❌", "❌"
   ":doc:`nixl_perftest`", "✅", "❌", "❌", "❌"
   ":doc:`sleep`", "✅", "✅", "❌", "✅"
   ":doc:`sglang`", "✅", "❌", "❌", "❌"
   ":doc:`srt_slurm`", "✅", "❌", "❌", "❌"
   ":doc:`slurm_container`", "✅", "❌", "❌", "❌"
   "Triton Inference", "✅", "❌", "❌", "❌"
   ":doc:`ucc`", "✅", "❌", "❌", "❌"
   ":doc:`vllm`", "✅", "❌", "❌", "❌"


System availability and workload support are distinct. See :doc:`/systems` for system configuration.
