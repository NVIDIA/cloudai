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

Choose a benchmark
------------------

.. tab-set::

   .. tab-item:: Inference
      :selected:

      .. _inference-with-vllm:

      Benchmark Qwen3-0.6B on one GPU on a Slurm node.

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

      .. code-block:: bash

         uv run cloudai run --system-config system.toml \
             --test-scenario conf/experimental/quickstart/nccl.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/nccl.toml
            :language: toml
            :start-at: name =

      :doc:`NCCL guide <workloads/nccl>`

   .. tab-item:: Storage

      .. _storage-with-fio:

      Install ``fio`` locally. Set ``filename`` to a disposable file on the storage to benchmark
      (default: ``/tmp/cloudai-fio-quickstart``, 80 MiB).

      .. code-block:: bash

         uv run cloudai run --system-config conf/common/system/standalone_system.toml \
             --test-scenario conf/experimental/quickstart/fio.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/fio.toml
            :language: toml
            :start-at: name =

      :doc:`Fio guide <workloads/fio>`

   .. tab-item:: Training

      .. _training-with-megatron-bridge:

      Run ten Qwen3-30B-A3B training steps on two eight-GPU H100 Slurm nodes.
      Set ``HF_TOKEN`` and prepare the :doc:`Megatron Bridge prerequisites <workloads/megatron_bridge>`.

      .. code-block:: bash

         uv run cloudai run --system-config system.toml \
             --test-scenario conf/experimental/quickstart/mbridge.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/mbridge.toml
            :language: toml
            :start-at: name =

      :doc:`Megatron Bridge guide <workloads/megatron_bridge>`

.. _getting-started:

Get started
-----------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ 0.13.4 or newer, then clone CloudAI:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloudai.git
   cd cloudai

For Slurm, copy ``conf/common/system/example_slurm_cluster.toml`` to ``system.toml`` and set your partitions,
nodes, and GPU counts. Prepare the :doc:`workload prerequisites <workloads/workloads_requirements_installation>`.
See :doc:`getting-started` for installation alternatives.

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
