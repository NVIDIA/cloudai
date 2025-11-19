AI Dynamo
=========

This workload (`test_template_name` is ``AIDynamo``) runs AI inference benchmarks using the Dynamo framework with distributed prefill and decode workers.


Usage Example
-------------

See :doc:`../ai_dynamo` for details.

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.ai_dynamo.ai_dynamo.AIDynamoCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.ai_dynamo.ai_dynamo.AIDynamoTestDefinition
   :members:
   :show-inheritance:

Run using Kubernetes
--------------------

Prepare cluster
~~~~~~~~~~~~~~~
Before running the AI Dynamo workload on a Kubernetes cluster, ensure that the cluster is set up according to the instructions in the `official documentation`_. Below is a short summary of the required steps:

.. _official documentation: https://docs.nvidia.com/dynamo/latest/_sections/k8s_deployment.html

.. code-block:: bash

   export NAMESPACE=dynamo-system
   export RELEASE_VERSION=0.6.1  # replace with the desired release version

   helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
   helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

   helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
   helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace


Run CloudAI to deploy AI Dynamo worker nodes according to your spec and run ``genai-perf`` tests:

.. code-block:: bash

   uv run cloudai run --system-config <k8s system toml> \
      --tests-dir conf/staging/ai_dynamo/test \
      --test-scenario conf/staging/ai_dynamo/test_scenario/vllm_k8s.toml