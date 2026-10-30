SRT-Slurm
=========

The ``SRTSlurm`` workload delegates benchmark job generation and submission to
srt-slurm through ``srtctl apply``, using its recipe and cluster configuration.
Slurm schedules the job, while CloudAI monitors completion and processes results.
The pinned srtctl version validates the recipe, including recipes without a
``schema`` field. Each CloudAI trial
passes selected ``recipe_overrides`` as typed ``--set KEY=VALUE`` arguments;
the original YAML remains unchanged.

Configuration
-------------

Use ``conf/experimental/srt_slurm/scenario.toml`` as a starting point. Adapt
its cluster configuration to your system. Omit ``srtctl_root`` to let SRT derive
the installed checkout path. Paths are resolved from the CloudAI working directory
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

The repository named ``srt-slurm`` in ``git_repos`` supplies srtctl, regardless
of entry order. Forks with that repository name are supported; other entries use
ordinary CloudAI installation. Pin a revision supporting ``srtctl apply --set``.
For older revisions without ``--set``, set ``cmd_args.recipe_override_mode = "render"``.
This overlays existing mapping keys in a concrete recipe, preserves YAML value types,
and submits a trial-local ``srt_slurm_recipe.yaml``. It rejects missing keys
and does not resolve recipe variants or interpret native override path expressions;
select the effective concrete recipe first. The default remains ``"native"``.
The launch script ``run_srtctl.sh`` and rendered recipe are stored directly in
each trial's output directory, alongside ``run_srtctl.log`` and ``srt_job_id.txt``.
SRT Slurm job outputs are stored under ``srt_slurm_outputs/`` in that directory.
Lists in ``recipe_overrides`` are DSE dimensions. Use nested candidate lists
when the selected recipe value is itself a list. Recipe paths, cluster config
paths, and result format are excluded from DSE. Additional apply flags use
``extra_cmd_args``, for example ``extra_cmd_args = { no-preflight = "" }``.

``cluster_config`` is required and takes precedence over ``SRTSLURM_CONFIG``
in the submitting environment. Supply a complete config with the recipe's
aliases, mounts, and cluster settings. CloudAI exports merged system and test
environment variables, removes virtual-environment directories from ``PATH``,
unsets ``VIRTUAL_ENV``, and submits from the installed srt-slurm checkout. These
steps prevent compute-node ``uv`` from selecting login-node Python through its
PATH search, active-environment lookup, or current-directory ``.venv`` lookup.
Tools supplied through virtual-environment PATH entries are removed as part of
this compatibility cleanup; supply required compute-node tools outside a Python
virtual environment.

CloudAI also exports ``CLOUDAI_INSTALL_FOLDER`` as its absolute ``install_path``.
SRT cluster configs can use its existing environment expansion in mount sources,
for example ``${CLOUDAI_INSTALL_FOLDER}/InferenceX__<commit>: /infmax-workspace``.
This keeps the scenario independent of the installation root; the checkout name
must still match the configured repository and commit.

Installation runs ``make setup`` in a single-node, 15-minute ``srun`` using
``system.toml``'s ``default_partition``, ``account``, and ``extra_srun_args``.
These settings apply only to the installation allocation; benchmark scheduling
uses the supplied srt-slurm cluster config and recipe. Setup downloads runtime
tools for the compute-node architecture. A checkout-local bootstrap
config prevents interactive setup prompts; existing checkout configs are
preserved. Trial submission always uses the explicit ``cluster_config``.

Model installation
~~~~~~~~~~~~~~~~~~

Optionally set ``hf_models = ["Qwen/Qwen3-0.6B"]`` on the CloudAI test definition
(outside ``cmd_args``) to download models through the existing ``HFModel``
installer. The default empty list preserves support for pre-staged weights.
``system.toml``'s ``hf_home_path`` controls the cache root; it defaults to
``<install_path>/huggingface``. Downloads are stored under
``<hf_home_path>/hub/models--<owner>--<model>/snapshots/<revision>``.

HF cache snapshots contain relative links to ``../../blobs``. When SRT mounts a
snapshot at ``/model``, also mount that model cache's ``blobs`` directory at
``/blobs`` through ``srtslurm.yaml``'s ``default_mounts``. The snapshot and blob
mount sources must use the same ``hf_home_path`` cache root. See the AgentX DSE
example for a complete configuration.

Set the corresponding ``model_paths`` alias in ``srtslurm.yaml`` to that snapshot
directory. Its cache root must match ``hf_home_path`` and be accessible on the
submission and compute nodes. CloudAI does not infer models from recipes or
synchronize these paths. To pin a model commit, use
``hf_models = [{model_name = "Qwen/Qwen3-0.6B", revision = "<commit>"}]`` and put
the same commit in the SRT snapshot path. String entries remain unpinned.
Pinned models are checked and uninstalled by revision; uninstalling an unpinned
model retains the existing behavior of removing the entire cached model. If the HF
cache is inaccessible locally, the installer assumes models are already available
on execution nodes instead of downloading them.

Scheduling ownership
~~~~~~~~~~~~~~~~~~~~

Configure benchmark scheduling in the supplied ``cluster_config``
(``srtslurm.yaml``) and recipe. CloudAI does not translate or merge its system
or scenario allocation settings into these files or into ``srtctl`` scheduling
arguments. Recipe overrides explicitly selected through ``recipe_overrides``
are still passed to srt-slurm as described above.

* ``srtslurm.yaml`` supplies cluster defaults such as account, partition, time
  limit, GPU capacity, and supported SBATCH directives.
* The recipe specifies serving roles, nodes, workers, GPU requirements,
  container mounts, profiling, and setup/cleanup commands, together with any
  recipe-specific scheduling overrides. srt-slurm resolves these inputs.
* CloudAI's ``system.toml`` selects the Slurm runner and supplies installation,
  cache, and output locations, shared environment variables, and monitoring
  settings. Its partition, account, and extra srun arguments are used for the
  installation setup allocation described above, not forwarded to benchmark
  submission.

For example, changing ``default_partition`` or ``account`` in CloudAI's system
config does not change the benchmark's partition or account. Likewise,
``extra_sbatch_args``, ``ntasks_per_node``, and scenario allocation settings
such as ``num_nodes``, ``nodes``, ``exclude_nodes``, and ``time_limit`` are not
translated into the benchmark allocation. Set the corresponding values in
srt-slurm's inputs. GPU capacity for benchmark scheduling comes from those
inputs, not CloudAI's ``gpus_per_node``.

Keep ``scheduler = "slurm"`` in the CloudAI system config: the Slurm runner
tracks the job ID returned by srt-slurm, monitors completion, handles
cancellation, and collects accounting information. Using this runner does not
make CloudAI the owner of the benchmark allocation.

When changing clusters, select both the target CloudAI system config and its
matching ``cluster_config`` in the scenario, and check the recipe for explicit
cluster-specific paths or scheduling overrides.

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
DSE topology. CSV result rows include ``serving_gpus``, ``aggregate_gpus``,
``prefill_gpus``, and ``decode_gpus``.

* ``default`` and ``output_throughput`` expose output tokens/s.
* ``total_token_throughput`` exposes input plus output tokens/s.
* ``tput_per_gpu`` divides total throughput by the metadata-derived serving count.
  Disaggregated input/output per-GPU metrics use their respective prefill/decode
  pools; aggregated metrics share the aggregate pool, counted once.
* Producer-normalized rollup fields are ignored. A rollup lacking raw total
  throughput cannot provide ``tput_per_gpu``.
* Latency aliases such as ``median_ttft`` use seconds; ``ttft_median_ms`` uses
  milliseconds.
* ``median_intvty`` and ``p90_intvty`` use the reciprocal of the corresponding
  TPOT, in tokens/s/user.

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
