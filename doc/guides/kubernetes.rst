Run on Kubernetes
=================

Run Dynamo inference or NCCL communication experiments on an existing GPU cluster.
CloudAI submits Kubernetes resources, monitors execution, and saves results locally.

Prepare cluster access
----------------------

* Install CloudAI using :doc:`/getting-started` and make ``kubectl`` available on the machine running CloudAI.
* Use a kubeconfig with permission to create, inspect, and delete workload resources in your namespace.
* Ensure GPU resources are available to pods and the cluster can pull the example's public container image.

Save this as ``kubernetes.toml``. Replace the kubeconfig path, namespace, and GPU count for your cluster:

.. code-block:: toml

   name = "kubernetes-cluster"
   scheduler = "kubernetes"
   kube_config_path = "/absolute/path/to/kubeconfig"
   default_namespace = "cloudai-experiments"
   gpus_per_node = 8
   install_path = "./install"
   output_path = "./results"

Point ``kubectl`` at the same kubeconfig and current context as CloudAI:

.. code-block:: bash

   export KUBECONFIG=/absolute/path/to/kubeconfig
   kubectl get nodes
   kubectl get namespace cloudai-experiments

Use an existing namespace or create it before running an experiment.

Choose an experiment
--------------------

.. list-table::
   :header-rows: 1

   * - Example
     - Cluster prerequisite
     - CloudAI submits
   * - :ref:`Dynamo inference <dynamo-kubernetes>`
     - Dynamo operator and platform services
     - ``DynamoGraphDeployment`` (``nvidia.com/v1alpha1``)
   * - :ref:`NCCL all-reduce <nccl-kubernetes>`
     - Kubeflow MPI Operator
     - ``MPIJob`` (``kubeflow.org/v2beta1``)

Sleep also supports Kubernetes for scheduling checks. See :doc:`/reference/workload-support` for the complete matrix.

.. _kubernetes-networking:

Use Network Operator attachments
--------------------------------

For Dynamo and NCCL, CloudAI can attach worker pods to networks provisioned by
`NVIDIA Network Operator <https://docs.nvidia.com/networking/display/kubernetes2610/quick-start/quick-start-k8s.html>`_.
Prepare the operator, device resources, and NetworkAttachmentDefinitions on the cluster first.

Add this top-level setting to ``kubernetes.toml`` to require network attachments:

.. code-block:: toml

   use_host_network = false

Check the attachments CloudAI will discover:

.. code-block:: bash

   kubectl get network-attachment-definitions --all-namespaces

CloudAI uses **all discovered attachments**, adds their ``namespace/name`` values to the worker pod annotations,
and requests one ``nvidia.com/<attachment-name>`` device resource per attachment.
For example, ``networks/rdma0`` must have a corresponding ``nvidia.com/rdma0`` resource on eligible nodes.
Ensure every discovered attachment is usable by your workload and follows this naming convention.

.. list-table::
   :header-rows: 1

   * - ``use_host_network``
     - Behavior for Dynamo and NCCL
   * - Omitted
     - Discover attachments; fall back to host networking if none are returned.
   * - ``false``
     - Require attachments; fail if discovery returns none.
   * - ``true``
     - Use host networking and skip attachment discovery.

Discovery uses ``kubectl`` across all namespaces; the kubeconfig needs permission to list those resources.
The generated ``deployment.yaml`` in each case's output directory shows the selected networking and device requests.

Inspect results
---------------

CloudAI saves the generated deployment, resolved test configuration, and workload output under ``output_path``.
It removes the submitted workload resources when the experiment completes. Use :doc:`/reporting` to inspect results
and :ref:`kubernetes-system` for the system field reference.
