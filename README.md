# CloudAI

**The execution framework for AI-driven research.**

Run the workloads your task requires, collect evidence, and use the results to guide the next experiment.
Combine experiments into a broader investigation using existing workloads or your own.

![CloudAI connects experiment configuration to workload execution, metrics, and reusable results. Custom agents can use measured feedback. Public interfaces support workloads, systems and runners, reporters, and agents.](doc/_static/cloudai-experiments.svg)

- **Run different workloads through one framework.** Use a common CLI, Python API, and configuration model.
- **Search for better configurations.** Evaluate a grid of choices or integrate an optimization agent that selects experiments and controls stopping within a compute budget.
- **Add what your task needs.** Extend workloads, execution systems, reporters, and agents through public interfaces and the registry.

## Run your first experiment

Install `uv` 0.13.4 or newer, then clone CloudAI:

```bash
git clone https://github.com/NVIDIA/cloudai.git
cd cloudai
```

Choose an example below. Fio runs locally without a GPU; inference, networking, and training use Slurm.
More integrations are listed in the [workload catalog](doc/workloads/index.rst).

<details>
<summary>Storage — run Fio locally</summary>

Install the Fio executable (`sudo apt-get install fio` on Ubuntu), then run:

```bash
cp conf/common/system/standalone_system.toml standalone.toml
uv run cloudai run --system-config standalone.toml \
    --test-scenario conf/experimental/quickstart/fio.toml
```

This writes an 80 MiB file at `/tmp/cloudai-fio-quickstart` using random writes.
To measure another filesystem, change `filename` in the [scenario](conf/experimental/quickstart/fio.toml).
Adjust `rw`, `bs`, and `size` for your I/O pattern.

</details>

<details>
<summary>Slurm setup — required for the GPU examples</summary>

Run CloudAI from a submit host with Slurm CLI tools, partition access, and container support for `srun`
(such as Pyxis/Enroot). Execution nodes must be able to access the selected public images and shared files.

Save this as `slurm.toml` in the repository root:

```toml
name = "my-cluster"
scheduler = "slurm"
install_path = "./install"
output_path = "./results"
default_partition = "YOUR_PARTITION"
mpi = "pmix"
gpus_per_node = 8
ntasks_per_node = 8

[[partitions]]
name = "YOUR_PARTITION"
```

- Replace both `YOUR_PARTITION` values and set GPU/task counts for your nodes.
- Add `account` before `[[partitions]]` if your cluster requires one.
- Set `mpi` to the MPI plugin available on your cluster.
- Use shared absolute install/output paths if compute nodes cannot access the repository directory.

CloudAI installs declared workload prerequisites during execution. Cluster container support and required model access must already be configured.

</details>

<details>
<summary>Inference — vLLM on one GPU</summary>

After the Slurm setup above, benchmark Qwen3-0.6B on one GPU:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/vllm.toml
```

The [scenario](conf/experimental/quickstart/vllm.toml) sends 30 prompts at concurrency 16,
with 16 input and 128 output tokens. Change the model, traffic, or parallelism settings to match your task.

</details>

<details>
<summary>Networking — NCCL across two nodes</summary>

After the Slurm setup above, measure all-reduce across two nodes:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/nccl.toml
```

The [scenario](conf/experimental/quickstart/nccl.toml) measures message sizes from 8 MiB to 128 MiB.
Change the collective, message sizes, or node count to investigate another communication pattern.

</details>

<details>
<summary>Training — Megatron Bridge on two eight-GPU H100 nodes</summary>

Complete the Slurm setup above and set `HF_TOKEN` in your environment. Run ten Qwen3-30B-A3B training steps:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/mbridge.toml
```

The [scenario](conf/experimental/quickstart/mbridge.toml) specifies the model recipe, GPU layout,
container, and training duration. Start with a short run before increasing the duration or scale.

</details>

## Use the results

Open the results directory printed by CloudAI:

- **`experiment.json`** — execution status, metrics, and selected search results.
- **HTML reports** — measurements and comparisons for inspection.
- **`stdout.txt` and `stderr.txt`** — workload output for diagnosing failures or missing metrics.

Keep model, input, resource, and measurement conditions consistent when comparing configurations.

<details>
<summary>Search configurations — compare two NCCL algorithms</summary>

After running the networking example, compare Ring and Tree using the same Slurm setup:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/nccl-search.toml
```

The [search scenario](conf/experimental/quickstart/nccl-search.toml) adds these settings to the case:

```toml
agent = "grid_search"
agent_metrics = ["default"]
agent_reward_function = "inverse"

[Tests.extra_env_vars]
NCCL_ALGO = ["Ring", "Tree"]
```

Grid search evaluates every configured combination. This example selects the successful step with the highest inverse
mean in-place latency reward. Inspect `tests[].dse` and `tests[].metrics` in `experiment.json`, the DSE report,
and the selected configuration.

Additional dimensions multiply the candidate count. Custom agents can use measured feedback to choose subsequent
configurations and stop according to their own policy.

</details>

<details>
<summary>Automate experiments with Python</summary>

After the local Fio setup, save this as `experiment.py` and run `uv run python experiment.py`:

```python
from pathlib import Path

import cloudai.api

system = Path("standalone.toml")
scenario = Path("conf/experimental/quickstart/fio.toml")
cloudai.api.validate_scenario(scenario, system)
experiment = cloudai.api.run_experiment(scenario, system)
print(experiment.status, experiment.path)
```

Execution waits for completion. Workload failures appear in the returned status;
configuration and setup errors raise exceptions.

</details>

## Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) to propose changes or add an integration.

## License

CloudAI is licensed under [Apache 2.0](LICENSE.md).

## External Materials

See the [Notice and Disclaimer](https://nvidia.github.io/cloudai/#external-materials-notice) in the CloudAI documentation before using external materials retrieved by CloudAI.
