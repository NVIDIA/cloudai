.. _cloudai-benchmark-framework:

CloudAI
========

**The execution framework for AI-driven research.**

Run the workloads your task requires, collect evidence, and use the results to guide the next experiment.
Combine experiments into a broader investigation using existing workloads or your own.

.. _overview:
.. _configuration-search:

.. image:: _static/cloudai-experiments.svg
   :alt: CloudAI connects experiment configuration to workload execution, metrics, and reusable results. Custom agents can use measured feedback. Public interfaces support workloads, systems and runners, reporters, and agents.
   :target: _static/cloudai-experiments.svg
   :width: 100%
   :class: cloudai-workflow dark-light

Explore configurations with grid search, or connect your own `optimization agent <USER_GUIDE.html#agents-configuration>`_
to guide search within your compute budget.
:doc:`Add workloads, systems, and reporters <DEV>` for your task.

.. _choose-a-workflow:

Run your first experiment
-------------------------

Start with an example below, or explore :doc:`all supported workloads <workloads/index>`.

.. tab-set::
   :sync-group: benchmark

   .. tab-item:: Inference
      :selected:
      :sync: inference

      .. _inference-with-vllm:

      Benchmark Qwen3-0.6B on one GPU on a Slurm node.

      :ref:`Slurm setup <quickstart-slurm>`

      .. code-block:: bash

         uv run cloudai run --system-config slurm.toml \
             --test-scenario conf/experimental/quickstart/vllm.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/vllm.toml
            :language: toml
            :start-at: name =

      :doc:`vLLM guide <workloads/vllm>`

   .. tab-item:: Networking
      :sync: networking

      .. _networking-with-nccl:

      Benchmark all-reduce on two Slurm nodes.

      :ref:`Slurm setup <quickstart-slurm>`

      .. code-block:: bash

         uv run cloudai run --system-config slurm.toml \
             --test-scenario conf/experimental/quickstart/nccl.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/nccl.toml
            :language: toml
            :start-at: name =

      :doc:`NCCL guide <workloads/nccl>`

   .. tab-item:: Storage
      :sync: storage

      .. _storage-with-fio:

      Benchmark file I/O with Fio. Set the test file path to the storage you want to measure.

      **Standalone** · :ref:`Setup <quickstart-standalone>`

      .. code-block:: bash

         uv run cloudai run --system-config standalone.toml \
             --test-scenario conf/experimental/quickstart/fio.toml

      **Slurm** · :ref:`Setup <quickstart-slurm>`

      .. code-block:: bash

         uv run cloudai run --system-config slurm.toml \
             --test-scenario conf/experimental/quickstart/fio-slurm.toml

      .. dropdown:: View configurations

         .. rubric:: Standalone

         .. literalinclude:: ../conf/experimental/quickstart/fio.toml
            :language: toml
            :start-at: name =

         .. rubric:: Slurm

         .. literalinclude:: ../conf/experimental/quickstart/fio-slurm.toml
            :language: toml
            :start-at: name =

      :doc:`Fio guide <workloads/fio>` · `Public container <https://hub.docker.com/r/openeuler/fio>`_

   .. tab-item:: Training
      :sync: training

      .. _training-with-megatron-bridge:

      Run ten Qwen3-30B-A3B training steps on two eight-GPU H100 Slurm nodes.

      Set ``HF_TOKEN`` in your environment.

      :ref:`Slurm setup <quickstart-slurm>`

      .. code-block:: bash

         uv run cloudai run --system-config slurm.toml \
             --test-scenario conf/experimental/quickstart/mbridge.toml

      .. dropdown:: View configuration

         .. literalinclude:: ../conf/experimental/quickstart/mbridge.toml
            :language: toml
            :start-at: name =

      :doc:`Megatron Bridge guide <workloads/megatron_bridge>`

.. _getting-started:

Start here
----------

Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_ 0.13.4 or newer and clone CloudAI:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cloudai.git
   cd cloudai

.. _quickstart-standalone:

Standalone
~~~~~~~~~~

Run Fio locally with ``fio`` installed.

.. code-block:: bash

   cp conf/common/system/standalone_system.toml standalone.toml

On Ubuntu, install Fio with ``sudo apt-get install fio``.

.. _quickstart-slurm:

Slurm
~~~~~

On your Slurm submit host, prepare the :doc:`container prerequisites <workloads/workloads_requirements_installation>`
and copy the system configuration:

.. code-block:: bash

   cp conf/common/system/example_slurm_cluster.toml slurm.toml

Set your partitions, nodes, and GPU counts in ``slurm.toml``.
For Fio, set the host path in ``extra_container_mounts`` to the storage you want to measure.
See :doc:`systems` for system settings and :doc:`getting-started` for installation alternatives.

Read your results
-----------------

Open the results directory printed by CloudAI. See :doc:`reporting` for HTML reports and ``experiment.json``.
Use the :doc:`Python API <api>` to automate experiments and consume results.

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
