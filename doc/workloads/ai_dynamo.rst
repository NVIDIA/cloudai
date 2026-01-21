AI Dynamo
=========

This workload (`test_template_name` is ``AIDynamo``) runs AI inference benchmarks using the Dynamo framework with distributed prefill and decode workers.


Run using Kubernetes
--------------------

Prepare cluster
~~~~~~~~~~~~~~~
Before running the AI Dynamo workload on a Kubernetes cluster, ensure that the cluster is set up according to the instructions in the `official documentation`_. Below is a short summary of the required steps:

.. _official documentation: https://docs.nvidia.com/dynamo/latest/_sections/k8s_deployment.html

.. code-block:: bash

   export NAMESPACE=dynamo-system
   export RELEASE_VERSION=0.7.0  # replace with the desired release version

   helm upgrade -n default -i dynamo-crds https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
   helm upgrade -n default -i dynamo-platform https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz

   # The following components are required for multi node only.
   # Versions should be aligned with Dynamo version.
   helm upgrade -n default -i grove oci://ghcr.io/ai-dynamo/grove/grove-charts:v0.0.0-gd462e65
   helm upgrade -n default -i kai-scheduler oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler:0.0.0-4c29820

Launch and Monitor the Job
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::

   Both CloudAI and Dynamo will try to access HuggingFace Hub. To avoid ``429 Too Many Requests`` errors and access models under auth, it is recommended to define ``HF_TOKEN`` environment variable before invoking CloudAI. Once set, run ``uv run hf auth login`` to authenticate.

.. code-block:: bash

   uv run cloudai run --system-config <k8s system toml> \
      --tests-dir conf/experimental/ai_dynamo/test \
      --test-scenario conf/experimental/ai_dynamo/test_scenario/vllm_k8s.toml

Run Using Slurm
---------------

Node Configuration for AI Dynamo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AI Dynamo jobs use three distinct types of nodes:

- **Frontend node**: Hosts the coordination services (`etcd`, `nats`), the **frontend server**, the **request generator** (`genai-perf`), and the first decode worker
- **Prefill node(s)**: Handle the prefill stage of inference
- **Decode node(s)**: Handle the decode stage of inference (optional, depending on model and setup)

The total number of required nodes must be:

::

   num_prefill_nodes + num_decode_nodes

If there is a mismatch in the number of nodes between the schema and the test scenario, CloudAI will use the number of nodes specified in the test schema, ignoring the value in the test scenario.

All node role assignments and orchestration are automatically managed by CloudAI.

Launch and Monitor the Job
~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the job:

.. code-block:: bash

   uv run cloudai run --system-config <slurm system toml> \
      --tests-dir conf/experimental/ai_dynamo/test \
      --test-scenario conf/experimental/ai_dynamo/test_scenario/vllm_slurm.toml

One can monitor job progress using either of the following options:

.. code-block:: bash

   watch squeue --me

.. code-block:: bash

   watch tail -n 4 ./results/<scenario name>/*.txt

The frontend node will initially wait to allow weight loading on all nodes. Once ready, it will launch ``genai-perf``, which begins generating requests to the frontend server. All servers cooperate to complete inference, and the output will appear in ``stdout.txt``.

Review genai-perf benchmark results
-----------------------------------

After job completion, CloudAI will place the output logs and result files in the designated results directory. To analyze performance metrics and validate inference outcomes:

- Navigate to the results directory (e.g., ``./results/...``)
- Most importantly, open the ``profile_genai_perf.csv`` file to examine the final benchmarking results

This CSV file includes detailed metrics collected by genai-perf, such as request latency, throughput, and system utilization statistics. Use this data to evaluate the model's performance and identify potential bottlenecks or optimization opportunities.

::

   Metric,avg,min,max,p99,p95,p90,p75,p50,p25
   Time To First Token (ms),"1,146.31",249.48,"3,485.23","3,457.97","3,349.56","3,215.06","1,330.93",640.07,286.52
   Time To Second Token (ms),26.05,0.00,133.51,96.12,36.56,34.88,34.35,33.55,1.78
   Request Latency (ms),"6,406.20","5,371.47","9,608.72","9,436.13","9,046.58","9,028.16","6,549.60","5,690.23","5,493.63"
   Inter Token Latency (ms),30.35,27.59,35.60,35.23,33.88,32.53,31.05,30.13,29.04
   Output Sequence Length (tokens),174.45,164.00,187.00,186.22,183.10,180.10,177.00,174.00,171.75
   Input Sequence Length (tokens),"3,000.05","2,999.00","3,001.00","3,001.00","3,001.00","3,000.00","3,000.00","3,000.00","3,000.00"

   Metric,Value
   Output Token Throughput (per sec),261.25
   Request Throughput (per sec),1.50
   Request Count (count),40.00


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
