NCCL
====

Measure collective communication across your selected GPUs and nodes.

Start with the :doc:`Slurm setup </guides/environment>`.
The starter uses two nodes for all-reduce.

.. code-block:: bash

   uv run cloudai run --system-config slurm.toml \
       --test-scenario conf/experimental/quickstart/nccl.toml

Inspect execution status and measurements using :doc:`/reporting`.

.. _nccl-kubernetes:

Run on Kubernetes
-----------------

Prepare ``kubernetes.toml`` using :doc:`/guides/kubernetes` and install the
`Kubeflow MPI Operator <https://github.com/kubeflow/mpi-operator>`_ with the ``kubeflow.org/v2beta1`` API.
The namespace must permit the privileged launcher and worker pods used by this integration.

Reuse the two-worker all-reduce scenario with the Kubernetes system:

.. code-block:: bash

   uv run cloudai run --system-config kubernetes.toml \
       --test-scenario conf/experimental/quickstart/nccl.toml

.. dropdown:: View the scenario

   .. literalinclude:: ../../conf/experimental/quickstart/nccl.toml
      :language: toml
      :start-at: name =

``num_nodes = 2`` creates two MPI worker pods. Set ``gpus_per_node`` in the system config to the GPUs per worker;
CloudAI launches one MPI process per GPU. Kubernetes controls pod placement.

The example uses the public NVIDIA PyTorch image, which includes ``all_reduce_perf_mpi``.
Workers install an SSH server if needed, so they require package repository access in that case.
For RDMA device requests and network attachments, follow :ref:`kubernetes-networking`.

Inspect ``deployment.yaml`` for the MPIJob and ``stdout.txt`` for the NCCL measurements in the case output directory.

Adapt the workload
------------------

This workload (`test_template_name` is ``NcclTest``) allows users to execute NCCL benchmarks within the CloudAI framework.

Usage Example
-------------

Test TOML example:

.. code-block:: toml

   name = "my_nccl_test"
   description = "Example NCCL test"
   test_template_name = "NcclTest"

   [cmd_args]
   docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"

Test Scenario example:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "nccl.1"
   num_nodes = 1
   time_limit = "00:05:00"

   test_name = "my_nccl_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "nccl.1"
   num_nodes = 1
   time_limit = "00:05:00"

   name = "my_nccl_test"
   description = "Example NCCL test"
   test_template_name = "NcclTest"

     [Tests.cmd_args]
     docker_image_url = "nvcr.io#nvidia/pytorch:25.06-py3"
     subtest_name = "all_reduce_perf_mpi"
     iters = 100

Configuration and API reference
-------------------------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nccl_test.nccl.NCCLCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nccl_test.nccl.NCCLTestDefinition
   :members:
   :show-inheritance:
