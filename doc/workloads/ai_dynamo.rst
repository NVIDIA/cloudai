AI Dynamo
=========

AI Dynamo workload (`test_template_name` is ``AIDynamo``) runs AI inference benchmarks using the Dynamo framework with distributed prefill and decode workers.


Run Using Kubernetes
--------------------

Prepare Cluster
~~~~~~~~~~~~~~~
Before running the AI Dynamo workload on a Kubernetes cluster, ensure that the cluster is set up according to the instructions in the `official documentation`_. Below is a short summary of the required steps:

.. _official documentation: https://docs.nvidia.com/dynamo/kubernetes-deployment/deployment-guide

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

- **Frontend node**: Hosts the coordination services (`etcd`, `nats`), the **frontend server**, the **request generator** (`aiperf` by default, configurable via ``workloads`` in the test TOML), and the first decode worker
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

The job progress monitoring can be done using either of the following options:

.. code-block:: bash

   watch squeue --me

.. code-block:: bash

   watch tail -n 4 ./results/<scenario name>/*.txt

The frontend node will initially wait to allow weight loading on all nodes. Once ready, it will launch the configured benchmark tool (``aiperf`` by default), which begins generating requests to the frontend server. All servers cooperate to complete inference, and the output will appear in ``stdout.txt``.

Choosing a Benchmark Tool
~~~~~~~~~~~~~~~~~~~~~~~~~

The benchmark tool is controlled by the ``workloads`` field in the test TOML. Set ``aiperf.sh`` to use AIPerf:

.. code-block:: toml

   [cmd_args]
   workloads = "aiperf.sh"   # uses aiperf, writes aiperf_report.csv

To use genai-perf, set:

.. code-block:: toml

   [cmd_args]
   workloads = "genai_perf.sh"   # uses genai-perf, writes genai_perf_report.csv

   [cmd_args.genai_perf]
   cmd = "genai-perf profile"
   extra-args = "--streaming --verbose -- -v --async"

     [cmd_args.genai_perf.args]
     endpoint-type = "chat"
     output-tokens-mean = 500
     request-count = 50

Semantic Degradation With AIPerf Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AIDynamo uses AIPerf accuracy mode as its semantic degradation signal. Enable it by adding AIPerf accuracy flags under
``[cmd_args.aiperf.args]`` and running the ``aiperf.sh`` workload:

.. code-block:: toml

   [cmd_args]
   workloads = "aiperf.sh"

   [cmd_args.aiperf]
   setup-cmd = "python -m pip install --break-system-packages --upgrade 'aiperf[accuracy]==0.6.0.post1'"

   [cmd_args.aiperf.args]
   accuracy-benchmark = "mmlu"
   accuracy-n-shots = 5
   accuracy-tasks = "abstract_algebra"
   concurrency = 10
   extra-inputs = '{"temperature":0,"stop":["\n"]}'
   num-requests = 100

When ``accuracy-benchmark`` is configured, CloudAI expects AIPerf to produce ``accuracy_results.csv`` and exposes the
``accuracy`` metric from its ``OVERALL`` row. The metric is reported as a 0.0-1.0 fraction. Keep synthetic prompt and
token-length flags out of this mode; the benchmark dataset should come from AIPerf's accuracy benchmark.

The ``setup-cmd`` field is optional. It is useful for Dynamo images that include ``aiperf`` without its accuracy extra;
CloudAI runs it immediately before launching ``aiperf profile``.

Review Benchmark Results
------------------------

After job completion, CloudAI places output logs and result files in the designated results directory. The result file name depends on the configured ``workloads`` field:

- ``aiperf.sh`` → ``aiperf_report.csv`` for performance mode, ``accuracy_results.csv`` for accuracy mode
- ``genai_perf.sh`` → ``genai_perf_report.csv``

If AIPerf accuracy mode is enabled, CloudAI copies ``aiperf_artifacts/accuracy_results.csv`` to ``accuracy_results.csv``
in the run output directory and marks the run failed if that file is not produced.

Navigate to ``./results/<scenario>/<test-id>/0/`` and open the CSV to examine performance metrics.

Example ``aiperf_report.csv``:

::

   Metric,avg,min,max,p25,p50,p75,p99,std
   Inter Token Latency (ms),2.81,2.66,2.88,2.79,2.83,2.84,2.87,0.04
   Time to First Token (ms),49.87,17.15,99.91,49.35,49.87,50.52,92.31,9.20
   Time to Second Token (ms),0.50,0.03,4.05,0.03,0.04,0.04,3.47,1.08
   Request Latency (ms),1652.30,1203.61,6433.87,1453.19,1462.99,1466.72,6431.16,976.18
   Output Sequence Length (tokens),498.06,410.00,501.00,500.00,500.00,500.00,501.00,12.62
   Input Sequence Length (tokens),300.00,300.00,300.00,300.00,300.00,300.00,300.00,0.00

   Metric,Value
   Output Token Throughput (tokens/sec),598.78
   Total Token Throughput (tokens/sec),962.32
   Request Throughput (requests/sec),1.20
   Request Count,50.00

Supported Backends
------------------

The following backends are available via the ``conf/experimental/ai_dynamo/test/`` directory:

- **vLLM** (``vllm.toml``) — use with ``test_scenario/vllm_slurm.toml``
- **sglang** (``sglang.toml``) — use with ``test_scenario/sglang_slurm.toml``

Both backends use ``aiperf`` as the default benchmark tool and support disaggregated prefill/decode.


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
