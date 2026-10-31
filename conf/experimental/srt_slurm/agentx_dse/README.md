# AgentX DSE smoke test

This example provisions pinned SRT Slurm and InferenceX repositories, downloads
Qwen3-0.6B through CloudAI's model installer, then runs
SGLang with Qwen3-0.6B on one serving GPU of a four-GPU GB200 node. Grid search
tries context lengths 2048 and 4096 and maximizes `p90_intvty`.

The client replays two public fixtures from InferenceX's pinned AIPerf submodule
for 60 seconds. It uses `--unsafe-override`; this is a correctness smoke test,
not a canonical SemiAnalysis submission or a performance comparison.

## Configure

All `/shared/...` paths and the `gpu` partition are placeholders:

- Copy the common [Slurm system template](../../../common/system/example_slurm_cluster.toml)
  to your own `system.toml`. Adapt its partition, node groups, GPU counts, and
  environment variables to your cluster.
- Put `scenario.toml`, `srtslurm.yaml`, `recipe.yaml`, and `agentx_client.sh` in
  `/shared/cloudai/example`, or update the scenario paths and the corresponding
  `/agentx-test` mount together.
- Set `install_path = "/shared/cloudai/install"` and
  `output_path = "/shared/cloudai/results"` in your system config. CloudAI exports
  its absolute installation root as `CLOUDAI_INSTALL_FOLDER`; SRT expands that
  variable in the InferenceX mount in `srtslurm.yaml`. SRT derives its own checkout
  root, so changing `install_path` does not require editing either repo path.
- Set `hf_home_path = "/shared/huggingface"` in your system config. The scenario
  declares Qwen3-0.6B and its commit in `hf_models`; `cloudai install` downloads that revision
  into that cache. In `srtslurm.yaml`, `model_paths.qwen` must point to the
  snapshot under the same cache root (`hf_home_path/hub/models--Qwen--Qwen3-0.6B/snapshots/<revision>`).
  The snapshot revision must match the scenario's model revision. Keep the
  accompanying `blobs` directory mounted at `/blobs`: SRT mounts the snapshot at
  `/model`, where its `../../blobs` symlinks need that target. Both YAML paths must
  use the same `hf_home_path` cache root. CloudAI does not synchronize these paths;
  the cache must be accessible on the submission and compute nodes.
- Provide a compatible SGLang container at `/shared/images/sglang.sqsh`. SRT
  mounts the model snapshot at `/model`; the client uses its tokenizer offline.
- Choose your partition in both system and SRT cluster configs. CloudAI uses
  its partition for the setup allocation; SRT generates and submits benchmark
  jobs using its recipe and cluster config, and Slurm schedules them.
- Configure the Slurm CLI environment and any site-required container settings.
  Dependencies require network access. uv caches and managed Python downloads
  use node-local `/tmp`; other application caches still need writable storage.

The `srt-slurm` repository entry supplies srtctl, regardless of entry order.
The InferenceX entry installs its pinned submodules. The `/infmax-workspace` mount
connects that checkout to the custom benchmark client. If the InferenceX commit
changes, update the checkout basename in `srtslurm.yaml` to match.

## Run

After adapting the paths, run from the directory containing the input files.
Pass the path to your adapted copy of the common system template:

```bash
mkdir -p /shared/cloudai/install /shared/cloudai/results /shared/huggingface
cloudai install --system-config /path/to/system.toml --test-scenario scenario.toml
cloudai run --system-config /path/to/system.toml --test-scenario scenario.toml
```

Each trial writes its SRT job artifacts and `srt_slurm_results.csv`. The DSE
iteration writes `trajectory.csv` and the selected `agentx.toml` configuration;
scenario and DSE HTML reports summarize the results.
