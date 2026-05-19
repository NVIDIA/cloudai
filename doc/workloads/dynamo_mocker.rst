DynamoMocker
============

DynamoMocker workload (``test_template_name`` is ``DynamoMocker``) runs GPU-free LLM inference simulation
using ``dynamo.mocker`` and ``dynamo.frontend`` from the `ai-dynamo <https://github.com/ai-dynamo/dynamo>`_
package, then benchmarks the stack with `aiperf <https://github.com/NVIDIA/aiperf>`_ or
`genai-perf <https://github.com/triton-inference-server/client/tree/main/src/c%2B%2B/perf_analyzer/genai-perf>`_.

It is a **Standalone** workload (no Slurm/Kubernetes/RunAI required).

Prerequisites
-------------

CloudAI automatically installs ``ai-dynamo``, ``aiperf``, and ``genai-perf`` into a managed Python virtual
environment on first run — no manual pip install is needed.

**The one manual prerequisite is** ``nats-server``. The ``dynamo.mocker`` runtime uses NATS as its event plane.
Install it from the `official releases <https://github.com/nats-io/nats-server/releases>`_ or via a package
manager, then make it available either on ``PATH`` or by setting ``nats_cmd`` in the test TOML to the full path:

.. code-block:: toml

   [cmd_args]
   nats_cmd = "/path/to/nats-server -js"

On many clusters ``nats-server`` is pre-installed by administrators and is already on ``PATH``, in which case
the default value ``"nats-server -js"`` works without modification.

An ``HF_TOKEN`` environment variable is required to download gated models from HuggingFace Hub. Set it before
running:

.. code-block:: bash

   export HF_TOKEN=<your_token>

Topologies
----------

The workload supports two disaggregation modes, configured via ``cmd_args.worker.disaggregation_mode``:

- **Combined** (``none``): a single ``dynamo.mocker`` process handles both prefill and decode. Controlled
  by ``cmd_args.worker.num_workers``.
- **Disaggregated** (``prefill_decode``): separate prefill and decode mocker instances, mirroring the
  production ``ai_dynamo`` topology. Instance counts are set via
  ``cmd_args.worker.prefill_worker.num_nodes`` and ``cmd_args.worker.decode_worker.num_nodes``.

Benchmark Tools
---------------

Select the benchmark tool with ``cmd_args.benchmark_tool``:

- ``"aiperf"`` (default in the provided TOML) — uses the ``aiperf`` profiler
- ``"genai_perf"`` — uses ``genai-perf profile``

Parameters for the active tool are configured under ``[cmd_args.aiperf]`` or ``[cmd_args.genai_perf]``.

Run Using Standalone
--------------------

.. note::

   Set ``HF_TOKEN`` before running to allow model download from HuggingFace Hub.

.. code-block:: bash

   uv run cloudai run \
     --system-config conf/experimental/dynamo_mocker/system/standalone_system.toml \
     --tests-dir conf/experimental/dynamo_mocker/test \
     --test-scenario conf/experimental/dynamo_mocker/test_scenario/dynamo_mocker.toml

CloudAI will:

1. Install ``ai-dynamo``, ``aiperf``, and ``genai-perf`` into a managed venv (first run only).
2. Write a wrapper script and launch ``dynamo_mocker.sh``.
3. Start ``nats-server``, ``dynamo.mocker`` (prefill and decode), and ``dynamo.frontend``.
4. Run the benchmark and write results to the output directory.

Review Benchmark Results
------------------------

After the run completes, results are placed in ``results/<scenario_name>/<test_id>/``:

- ``benchmark_report.csv`` — full per-request and aggregate metrics (throughput, latency percentiles, TTFT, ITL)
- ``stdout.txt`` / ``stderr.txt`` — orchestration log and process output
- ``dynamo_prefill_0.log``, ``dynamo_decode_0.log``, ``dynamo_frontend.log`` — per-component logs
- ``nats.log`` — NATS server log

Key summary metrics from ``benchmark_report.csv``:

::

   Metric,Value
   Output Token Throughput (tokens/sec),667.58
   Request Count,50.00
   Request Throughput (requests/sec),18.04

   Metric,avg,p50,p99
   Request Latency (ms),507.49,475.50,893.26
   Time to First Token (ms),77.82,71.99,137.26
   Inter Token Latency (ms),12.03,11.55,16.42

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.dynamo_mocker.dynamo_mocker.DynamoMockerCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.dynamo_mocker.dynamo_mocker.DynamoMockerTestDefinition
   :members:
   :show-inheritance:
