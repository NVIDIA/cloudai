Workload catalog and compatibility
==================================

Choose the integration that executes the workload your investigation needs. Start with its smallest example, then adapt its settings.

Check the compatibility table before choosing an execution system. The four :doc:`landing-page examples </index>`
are starting points; the catalog below contains the other integrations.

Training
--------

.. toctree::
   :maxdepth: 1

   megatron_bridge
   nemo_run
   ddlb

Inference
---------

.. toctree::
   :maxdepth: 1

   vllm
   sglang
   ai_dynamo
   srt_slurm

Networking and communication
----------------------------

.. toctree::
   :maxdepth: 1

   nccl
   ucc
   osu
   deepep
   nixl_bench
   nixl_ep
   nixl_kvbench
   nixl_perftest

Storage
-------

.. toctree::
   :maxdepth: 1

   fio

Simulation and prediction
-------------------------

.. toctree::
   :maxdepth: 1

   aiconfigurator
   dynamo_mocker
   chakra_replay

General-purpose execution
-------------------------

.. toctree::
   :maxdepth: 1

   bash_cmd
   slurm_container
   sleep

Legacy integrations
-------------------

.. toctree::
   :maxdepth: 1

   nemo_launcher
   jax_toolbox

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
