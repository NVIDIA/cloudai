.. _cloudai-benchmark-framework:

CloudAI
========

**Benchmark, optimize, and extend.**

Run training, inference, communication, and storage benchmarks through one framework.
Explore configurations with optimization agents, compare measured results, and add the integrations your systems need.

.. _overview:
.. _configuration-search:

.. image:: _static/cloudai-workflow.svg
   :alt: CloudAI connects experiment configuration to workload execution, metrics, and reusable results. Custom agents can use measured feedback. Public interfaces support workloads, systems and runners, reporters, and agents.
   :target: _static/cloudai-workflow.svg
   :width: 100%
   :class: cloudai-workflow dark-light

:doc:`Extend CloudAI <DEV>` · `Configure optimization agents <USER_GUIDE.html#agents-configuration>`_

.. _choose-a-workflow:
.. _getting-started:

Run a benchmark
---------------

.. tab-set::

   .. tab-item:: Inference
      :selected:

      .. _inference-with-vllm:

      Benchmark Qwen3-0.6B on one GPU on a Slurm node.

      .. dropdown:: Set up this benchmark

         .. include:: _includes/benchmark-setup.txt

      Use your cluster's configuration as ``system.toml``.

      .. code-block:: bash

         uv run cloudai run --system-config system.toml \
             --test-scenario conf/experimental/quickstart/vllm.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/vllm.toml
            :language: toml
            :start-at: name =

      :doc:`vLLM guide <workloads/vllm>`

   .. tab-item:: Networking

      .. _networking-with-nccl:

      Benchmark all-reduce on two Slurm nodes.

      .. dropdown:: Set up this benchmark

         .. include:: _includes/benchmark-setup.txt

      Use your cluster's configuration as ``system.toml``.

      .. code-block:: bash

         uv run cloudai run --system-config system.toml \
             --test-scenario conf/experimental/quickstart/nccl.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/nccl.toml
            :language: toml
            :start-at: name =

      :doc:`NCCL guide <workloads/nccl>`

   .. tab-item:: Storage

      .. _storage-with-nixl:

      Benchmark POSIX storage writes with NIXL on one Slurm node.

      .. dropdown:: Set up this benchmark

         .. include:: _includes/benchmark-setup.txt

         Set ``output_path`` in ``system.toml`` to a writable directory on the storage to measure.

      Use your cluster's configuration as ``system.toml``.

      .. code-block:: bash

         uv run cloudai run --system-config system.toml \
             --test-scenario conf/experimental/quickstart/nixl-storage.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/nixl-storage.toml
            :language: toml
            :start-at: name =

      :doc:`NIXL Bench guide <workloads/nixl_bench>`

   .. tab-item:: Training

      .. _training-with-megatron-bridge:

      Run ten Qwen3-30B-A3B training steps on two eight-GPU H100 Slurm nodes.

      .. dropdown:: Set up this benchmark

         .. include:: _includes/benchmark-setup.txt

         Set ``HF_TOKEN`` in your environment and prepare the :doc:`Megatron Bridge prerequisites <workloads/megatron_bridge>`.

      Use your cluster's configuration as ``system.toml``.

      .. code-block:: bash

         uv run cloudai run --system-config system.toml \
             --test-scenario conf/experimental/quickstart/mbridge.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/mbridge.toml
            :language: toml
            :start-at: name =

      :doc:`Megatron Bridge guide <workloads/megatron_bridge>`

Read your results
-----------------

Open the results directory printed by CloudAI. See :doc:`reporting` for HTML reports and ``experiment.json``.

.. toctree::
   :hidden:

   getting-started
   Tutorial
   USER_GUIDE
   api
   reporting
   systems
   workloads/index
   DEV
   troubleshooting

.. _external-materials-notice:

Notice and Disclaimer
---------------------

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.
Those retrieved materials are not distributed with this software and are governed solely by separate terms,
conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable
terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved
materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The
author makes no representations or warranties regarding any retrieved materials, and assumes no liability for
any losses, damages, liabilities or legal consequences from your use or inability to use this software or any
retrieved materials. Use this software and the retrieved materials at your own risk.
