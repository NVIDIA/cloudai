AI Dynamo
=========

AI Dynamo workload (`test_template_name` is ``AIDynamo``) runs AI inference benchmarks using the Dynamo framework with distributed prefill and decode workers.


Run Using Kubernetes
--------------------

Prepare Cluster
~~~~~~~~~~~~~~~
Before running the AI Dynamo workload on a Kubernetes cluster, ensure that the cluster is set up according to the instructions in the `official documentation`_. Below is a short summary of the required steps:

.. _official documentation: https://docs.nvidia.com/dynamo/dev/getting-started/kubernetes-deployment

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

By default, when ``num_nodes`` is omitted, CloudAI allocates separate nodes for prefill and decode workers:

::

   num_prefill_nodes + num_decode_nodes

Set top-level ``num_nodes`` explicitly to control the Slurm allocation. A value lower than
``num_prefill_nodes + num_decode_nodes`` enables shared-node disaggregated inference, where prefill and decode roles
run on the same allocated node(s) with separate GPU slices.

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

AIPerf Multi-Phase Runs
~~~~~~~~~~~~~~~~~~~~~~~

``cmd_args.aiperf`` is the base AIPerf config. ``cmd_args.aiperf_phases`` can run several AIPerf rounds against the
same live Dynamo stack. By default, CloudAI does not restart prefill, decode, or router processes between phases:

.. code-block:: toml

   dse_excluded_args = ["cmd_args.aiperf_phases"]

   [cmd_args.aiperf]
   health-check-between-phases = true
   between-phase-cmd = "true"  # default no-op

     [cmd_args.aiperf.args]
     request-count = 50
     server-metrics = "auto"

   [[cmd_args.aiperf_phases]]
   name = "round_1"
     [cmd_args.aiperf_phases.args]
     concurrency = 2

   [[cmd_args.aiperf_phases]]
   name = "round_2"
     [cmd_args.aiperf_phases.args]
     concurrency = 4

Single-phase runs keep the old artifact layout: ``aiperf_artifacts/``, ``aiperf.log``, and ``aiperf_report.csv``.
Multi-phase runs write per-phase artifacts/logs/reports and copy the last phase report to ``aiperf_report.csv`` for
existing report generation.

``between-phase-cmd`` is a bash command run after each non-final phase. The default is a no-op. Set it explicitly for
backend-specific cache cleanup, for example ``/cloudai_run_results/routerctl.sh restart`` if a test needs to restart the
Dynamo router between phases. ``health-check-between-phases`` probes the frontend after the command.

Comparison Reports
------------------

CloudAI provides the scenario-level ``ai_dynamo_comparison`` report. It compares AI Dynamo runs using the standard
LLM serving tables and charts for TTFT, TPOT, successful prompts, output-token throughput, TPS per user, and TPS per
GPU. When ``aiperf_accuracy`` is configured and its result is available, the report also compares model accuracy.
Both AIPerf and GenAI-Perf CSV output are supported.

AIPerf args are rendered as normal CLI flags. Multi-value AIPerf options should be passed with AIPerf CLI syntax, such
as ``server-metrics-formats = "csv,json,jsonl"`` or ``gpu-telemetry = "node1:9401,node2:9401"``. ``server-metrics =
"auto"`` expands to the frontend metrics endpoint, Dynamo worker metrics endpoints, and any CloudAI-started DCGM
exporters.

Propagating LMCache Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AIDynamo can pass an LMCache YAML config to the worker processes by setting ``LMCACHE_CONFIG_FILE`` inside the
container. This only propagates the LMCache configuration; the vLLM/SGLang runtime still needs to be launched with the
appropriate LMCache or KV-transfer connector for that image/version.

The preferred form is structured TOML under ``[cmd_args.lmcache]``. CloudAI converts that object to YAML in the
run output directory, mounts that directory as ``/cloudai_run_results``, and exports the generated file path as
``LMCACHE_CONFIG_FILE``:

.. code-block:: toml

   [cmd_args]
     [cmd_args.lmcache]
     chunk_size = 256
     local_cpu = true
     controller_pull_url = "{frontend_node}:8300"
     controller_reply_url = "{frontend_node}:8400"
     lmcache_worker_ports = [8788, 8789, 8790, 8791]
     max_local_cpu_size = 6.0
     nixl_buffer_size = 2079377920
     nixl_buffer_device = "cpu"

       [cmd_args.lmcache.extra_config]
       enable_nixl_storage = false
       nixl_backend = "POSIX"
       nixl_path = "{storage_cache_dir}"
       nixl_pool_size = 2048

For an example that uses test-in-scenario mode, see
``conf/experimental/ai_dynamo/test_scenario/vllm_lmcache.toml``. Because the test is fully defined inside the scenario,
``--tests-dir`` is not required when running that example:

.. code-block:: bash

   uv run cloudai run --system-config <slurm system toml> \
      --test-scenario conf/experimental/ai_dynamo/test_scenario/vllm_lmcache.toml

The example sets ``dse_excluded_args = ["cmd_args.lmcache.lmcache_worker_ports"]`` because
``lmcache_worker_ports`` is a list-valued LMCache setting, not a DSE sweep dimension. Other list-valued LMCache fields
can still be swept unless their ``cmd_args.`` path is also excluded.

Alternatively, mount your own LMCache YAML file with ``extra_container_mounts`` and set ``LMCACHE_CONFIG_FILE`` through
``extra_env_vars``:

.. code-block:: toml

   extra_container_mounts = ["/host/lmcache:/lmcache"]
   extra_env_vars = { LMCACHE_CONFIG_FILE = "/lmcache/config.yaml" }

For multi-node LMCache storage tests, any path referenced by the LMCache YAML, such as ``nixl_path`` for POSIX-backed
storage, must be visible and writable from every node that is expected to share cached data. A node-local path such as
``/tmp`` is suitable only for single-node smoke tests or configuration propagation checks.

LMCache YAML values can use runtime placeholders. CloudAI renders them inside the Slurm job before launching workers:
``{frontend_node}``, ``{frontend_ip}``, ``{results_dir}``, and ``{storage_cache_dir}``. Unknown placeholders fail the
run before worker processes start.

If the selected LMCache mode needs a controller, CloudAI can start one on the frontend node:

.. code-block:: toml

   [cmd_args.lmcache_controller]
   cmd = "lmcache_controller --host 0.0.0.0 --port 9000 --monitor-ports {\"pull\":8300,\"reply\":8400}"

This only launches the process. For disaggregated or multi-node runs, the LMCache YAML still needs controller addresses
that resolve to the frontend node from every worker. With the default controller monitor ports, use
``controller_pull_url = "{frontend_node}:8300"`` and ``controller_reply_url = "{frontend_node}:8400"``. The
``lmcache_worker_ports`` list must match the number of worker ranks.

Semantic Degradation With AIPerf Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AIDynamo uses AIPerf accuracy mode as its semantic degradation signal. Enable it with
``[cmd_args.aiperf_accuracy]``. This runs after the configured performance workload, so it can be used with either
``aiperf.sh`` or ``genai_perf.sh``:

.. code-block:: toml

   [cmd_args]
   workloads = "aiperf.sh"

   [cmd_args.aiperf]
     [cmd_args.aiperf.args]
     request-count = 50
     synthetic-input-tokens-mean = 300
     output-tokens-mean = 500
     concurrency = 2

   [cmd_args.aiperf_accuracy]
   entrypoint = "aiperf profile"
   setup-cmd = "python -m pip install --break-system-packages --upgrade aiperf==0.8.0"
   cli = '''
   --model {model}
   --url {url}
   --endpoint-type chat
   --streaming
   --artifact-dir {artifact_dir}
   --no-server-metrics
   --accuracy-benchmark mmlu
   --accuracy-n-shots 5
   --accuracy-tasks abstract_algebra
   --concurrency 10
   --extra-inputs '{"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
   --num-requests 100
   '''

When ``cmd_args.aiperf_accuracy`` is configured, CloudAI expects AIPerf to produce ``accuracy_results.csv`` and exposes
the ``accuracy`` metric from its ``OVERALL`` row. The metric is reported as a 0.0-1.0 fraction. Keep synthetic prompt
and token-length flags out of this mode; the benchmark dataset should come from AIPerf's accuracy benchmark.

The ``entrypoint`` and ``cli`` fields form the accuracy command. CloudAI expands ``{model}``, ``{url}``,
``{endpoint}``, ``{result_dir}``, and ``{artifact_dir}`` in ``cli`` before launching it. The ``setup-cmd`` field is
optional. It is useful for Dynamo images that include an older system ``aiperf`` build without the accuracy benchmark
plugins. The example upgrades the image-level ``aiperf`` before launching ``aiperf profile``.
MMLU is loaded from ``lighteval/mmlu``, so either allow Hugging Face dataset access or pre-cache that dataset before
running with ``HF_HUB_OFFLINE``/``HF_DATASETS_OFFLINE`` enabled.
For Qwen3 models, the example disables thinking mode so short MMLU answers can be parsed as choices.

Custom Accuracy Scripts
~~~~~~~~~~~~~~~~~~~~~~~

``cmd_args.aiperf_accuracy`` can also launch a custom mounted script instead of AIPerf. Mount the script or its parent
directory with ``extra_container_mounts`` and set ``entrypoint`` to the in-container command:

.. code-block:: toml

   extra_container_mounts = ["/host/custom_accuracy:/custom_accuracy"]

   [cmd_args.aiperf_accuracy]
   entrypoint = "python /custom_accuracy/dummy_accuracy.py"
   cli = "--model {model} --url {url} --endpoint {endpoint} --artifact-dir {artifact_dir} --prompt ping"

CloudAI expands placeholders in ``cli`` and runs ``entrypoint`` with that CLI string. The custom command must write
``accuracy_results.csv`` inside ``{artifact_dir}`` with an ``OVERALL`` row. CloudAI copies that file to the run output
directory and exposes the same ``accuracy`` metric as AIPerf accuracy mode.

Review Benchmark Results
------------------------

After job completion, CloudAI places output logs and result files in the designated results directory. The result file name depends on the configured ``workloads`` field:

- ``aiperf.sh`` → ``aiperf_report.csv``
- ``genai_perf.sh`` → ``genai_perf_report.csv``
- ``cmd_args.aiperf_accuracy`` → ``accuracy_results.csv``

If AIPerf accuracy mode is enabled, CloudAI copies ``aiperf_accuracy_artifacts/accuracy_results.csv`` to
``accuracy_results.csv`` in the run output directory and marks the run failed if that file is not produced.

Navigate to ``./results/<scenario>/<test-id>/0/`` and open the CSV to examine performance metrics.

Shared-Node Disaggregated Runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Slurm, set top-level ``num_nodes`` lower than the sum of ``prefill_worker.num-nodes`` and
``decode_worker.num-nodes`` to run both roles on the same allocated node(s). For example, ``num_nodes = 1`` with
``prefill_worker.num-nodes = 1`` and ``decode_worker.num-nodes = 1`` runs one prefill worker and one decode worker on
the same node. CloudAI assigns decode GPUs first and prefill GPUs after that based on each role's
``tensor-parallel-size * pipeline-parallel-size``. The combined role GPU count must fit on one node.

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
- **vLLM with LMCache config propagation** — use self-contained scenario ``test_scenario/vllm_lmcache.toml``
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
