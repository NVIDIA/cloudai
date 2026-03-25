Workloads
=========

This chapter contains automatically generated documentation for all CloudAI workloads. Each workload provides specific functionality for running different types of tests and benchmarks.

Available Workloads
-------------------

.. csv-table::
   :header: "Test", "Slurm", "Kubernetes", "RunAI", "Standalone"
   :widths: 40, 15, 15, 15, 15

   ":doc:`aiconfigurator`", "❌", "❌", "❌", "✅"
   ":doc:`ai_dynamo`", "✅", "✅", "❌", "❌"
   ":doc:`bash_cmd`", "✅", "❌", "❌", "❌"
   ":doc:`chakra_replay`", "✅", "❌", "❌", "❌"
   ":doc:`ddlb`", "✅", "❌", "❌", "❌"
   ":doc:`deepep`", "✅", "❌", "❌", "❌"
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
   ":doc:`slurm_container`", "✅", "❌", "❌", "❌"
   "Triton Inference", "✅", "❌", "❌", "❌"
   ":doc:`ucc`", "✅", "❌", "❌", "❌"
   ":doc:`vllm`", "✅", "❌", "❌", "❌"

.. toctree::
    :hidden:
    :glob:

    *

Adding New Workloads
---------------------

To add documentation for a new workload:

1. Add docstrings to your Python classes and methods.
2. Create a reStructuredText file in ``doc/workloads/`` (e.g., ``my_workload.rst``).
3. Add it to the table above.

The documentation will be automatically generated during the build process.
