SRT-Slurm
=========

The ``SRTSlurm`` workload submits an existing recipe with ``srtctl apply`` and
monitors the resulting Slurm job directly. The pinned srtctl version validates
the recipe, including recipes without a ``schema`` field. Each CloudAI trial
passes selected ``recipe_overrides`` as typed ``--set KEY=VALUE`` arguments;
the original YAML remains unchanged.

Configuration
-------------

Use ``conf/experimental/srt_slurm/scenario.toml`` as a starting point. Adapt
its cluster configuration to your system, including ``srtctl_root`` pointing
to the installed checkout. Paths are resolved from the CloudAI working directory
and must be accessible on the shared filesystem.

.. code-block:: toml

   name = "srt_sglang"
   description = "SGLang with srt-slurm"
   test_template_name = "SRTSlurm"
   agent_metrics = ["tput_per_gpu"]

   [[git_repos]]
   url = "https://github.com/NVIDIA/srt-slurm.git"
   commit = "v2.2.1"

   [cmd_args]
   recipe_path = "/shared/recipes/sglang.yaml"
   cluster_config = "/shared/config/srtslurm.yaml"
   result_format = "sa-bench"

   [cmd_args.recipe_overrides.benchmark]
   concurrencies = [[1, 4], [8, 16]]

The first ``git_repos`` entry supplies srtctl; additional entries use ordinary
CloudAI installation. Pin a revision supporting ``srtctl apply --set``.
For older revisions without ``--set``, set ``cmd_args.recipe_override_mode = "render"``.
This overlays existing mapping keys in a concrete recipe, preserves YAML value types,
and submits a trial-local ``cloudai_generated/recipe.yaml``. It rejects missing keys
and does not resolve recipe variants or interpret native override path expressions;
select the effective concrete recipe first. The default remains ``"native"``.
Lists in ``recipe_overrides`` are DSE dimensions. Use nested candidate lists
when the selected recipe value is itself a list. Recipe paths, cluster config
paths, and result format are excluded from DSE. Additional apply flags use
``extra_cmd_args``, for example ``extra_cmd_args = { no-preflight = "" }``.

``cluster_config`` is required and takes precedence over ``SRTSLURM_CONFIG``
in the submitting environment. Supply a complete config with the recipe's
aliases, mounts, and cluster settings. CloudAI exports merged system and test
environment variables, removes login-node virtual environments from ``PATH``,
unsets ``VIRTUAL_ENV``, and submits from the installed srt-slurm checkout.

Installation runs ``make setup`` in a single-node, 15-minute ``srun`` using the
CloudAI system's partition, account, and extra srun arguments. This downloads
runtime tools for the compute-node architecture. A checkout-local bootstrap
config prevents interactive setup prompts; existing checkout configs are
preserved. Trial submission always uses the explicit ``cluster_config``.

The recipe and cluster config own job resources and orchestration. Configure
nodes, time limits, container mounts, profiling, and setup/cleanup commands
there. CloudAI's scenario allocation directives do not rewrite the recipe.
Use normal per-trial Slurm CLI execution; single-sbatch, Slurm REST submission,
and CloudAI pre/post-test hooks are unsupported for this workload.

Results
-------

``result_format = "sa-bench"`` reads job-local ``sa-bench_*/results_*.json``
files, falling back to ``benchmark-rollup.json`` when raw results are absent.
Both result formats require the submitted job's ``<job_id>.json`` next to its
``logs`` directory. Its ``job_id`` must match the tracked job, and ``resources``
must contain nonnegative integer worker counts for ``agg``, ``prefill`` and
``decode``, plus positive integer ``gpus_per_<role>`` for each active role.
Missing, malformed, mismatched or incomplete metadata makes the trial unsuccessful;
there is no fallback to manual counts, filenames, node capacity or resource snapshots.
Use a pinned revision containing the resolved GPU metadata feature (introduced in
commit ``7858d3091d5e278ab56ebf0246df9ea913bc2477``, first released in ``v1.0.0``).

CloudAI calculates each role's GPU total as workers times GPUs per worker and sums
those totals for the serving count. This automatically follows each trial's effective
DSE topology. ``serving_gpus``, ``prefill_gpus`` and ``decode_gpus`` are no longer
scenario fields; remove them from existing configs. Resolved counts appear in each
CSV result row as ``serving_gpus``, ``aggregate_gpus``, ``prefill_gpus`` and ``decode_gpus``.

``default`` and ``output_throughput`` expose output tokens/s;
``total_token_throughput`` exposes input plus output tokens/s.
``tput_per_gpu`` divides total throughput by the metadata-derived serving count.
Disaggregated input/output per-GPU metrics use their respective prefill/decode pools;
aggregated metrics share the aggregate pool, counted once. Producer-normalized
rollup fields are ignored. A rollup lacking raw total throughput cannot provide
``tput_per_gpu``. Latency aliases such as ``median_ttft`` use seconds;
``ttft_median_ms`` uses milliseconds. ``median_intvty`` and ``p90_intvty``
use the reciprocal of the corresponding TPOT, in tokens/s/user.

``result_format = "agentx"`` reads InferenceX aggregate JSON from the submitted
job's ``logs/agentic`` directory. Each aggregate must have its corresponding
``conc_<c>/aiperf_artifacts/profile_export.jsonl``. CloudAI reads the existing
aggregation rather than recomputing per-request statistics. CloudAI recomputes
per-GPU throughput from raw ``request_metrics.throughput`` total/input/output values
using the job metadata's total serving count for all three streams. Producer
``per_gpu`` values are ignored, since environment-derived divisors can become stale
when DSE changes topology. An aggregate without raw total throughput is unusable.
Interactivity uses full-response ITL
when available, otherwise ITL. Latency aliases use the same units as sa-bench.

An AgentX recipe uses ``benchmark.type: custom`` and its existing InferenceX
command. Supply a pinned InferenceX checkout with ``init_submodules = true``
and the recipe's required ``INFMAX_WORKSPACE`` environment. CloudAI exports
``AGENTIC_OUTPUT_DIR=/logs/agentic`` for this result format. Keep that same value
in any recipe benchmark environment, and keep ``RESULT_DIR=/logs/agentic``
if explicitly configured, so aggregate and profile outputs remain job-local.

Reports write ``srt_slurm_results.csv`` with every concurrency. Scalar DSE
metrics come from one point: the highest per-GPU total throughput when available,
otherwise the highest output throughput. All latency metrics refer to that
same point. Missing, malformed, and nonfinite results do not mark a trial
successful; unavailable metrics return CloudAI's metric error value.
