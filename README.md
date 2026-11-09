# CloudAI

**The execution framework for AI-driven research.**

Run the workloads your task requires, collect evidence, and use the results to guide the next experiment.
Combine experiments into a broader investigation using existing workloads or your own.

![CloudAI connects experiment configuration to workload execution, metrics, and reusable results. Custom agents can use measured feedback. Public interfaces support workloads, systems and runners, reporters, and agents.](doc/_static/cloudai-experiments.svg)

[Extend CloudAI](https://nvidia.github.io/cloudai/DEV.html) ·
[Configure optimization agents](https://nvidia.github.io/cloudai/reference/agents.html)

## Run your first experiment

[Set up CloudAI](#get-started), then choose an example. Storage runs locally without a GPU;
the other examples use Slurm. [All supported workloads](https://nvidia.github.io/cloudai/workloads/index.html).

<details>
<summary>Inference — vLLM, one GPU</summary>

Benchmark Qwen3-0.6B on one GPU on a Slurm node. Use the Slurm setup below.

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/vllm.toml
```

<details>
<summary>View configuration</summary>

**Scenario:** [vllm.toml](conf/experimental/quickstart/vllm.toml)

```toml
name = "quickstart-vllm"

[[Tests]]
id = "vllm"
name = "vllm"
description = "Single-node Qwen inference benchmark"
test_template_name = "vllm"
num_nodes = 1
time_limit = "00:10:00"

  [Tests.cmd_args]
  docker_image_url = "nvcr.io#nvidia/ai-dynamo/vllm-runtime:1.1.1"
  model = "Qwen/Qwen3-0.6B"
  serve_healthcheck = "/health"

    [Tests.cmd_args.decode]
    gpu_ids = "0"
    tensor_parallel_size = 1

  [Tests.bench_cmd_args]
  random_input_len = 16
  random_output_len = 128
  max_concurrency = 16
  num_prompts = 30
```

</details>

</details>

<details>
<summary>Networking — NCCL, two nodes</summary>

Benchmark all-reduce on two Slurm nodes. Use the Slurm setup below, with GPU and task counts matching your nodes.

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/nccl.toml
```

<details>
<summary>View configuration</summary>

**Scenario:** [nccl.toml](conf/experimental/quickstart/nccl.toml)

```toml
name = "quickstart-nccl"

[[Tests]]
id = "all-reduce"
name = "all-reduce"
description = "Two-node NCCL all-reduce benchmark"
test_template_name = "NcclTest"
num_nodes = 2
time_limit = "00:05:00"

  [Tests.cmd_args]
  docker_image_url = "nvcr.io#nvidia/pytorch:25.12-py3"
  subtest_name = "all_reduce_perf_mpi"
  ngpus = 1
  minbytes = "8M"
  maxbytes = "128M"
  iters = 20
  warmup_iters = 5
  stepfactor = 2
```

</details>

</details>

<details>
<summary>Storage — Fio, local or Slurm</summary>

Measure file I/O with an 80 MiB random-write test. Set the filename to the storage you want to measure.

**Local:** use the Standalone setup below. The default test file is `/tmp/cloudai-fio-quickstart`.

```bash
uv run cloudai run --system-config standalone.toml \
    --test-scenario conf/experimental/quickstart/fio.toml
```

<details>
<summary>View configuration</summary>

**Scenario:** [fio.toml](conf/experimental/quickstart/fio.toml)

```toml
name = "quickstart-fio"

[[Tests]]
id = "fio"
name = "fio"
description = "Small file I/O benchmark"
test_template_name = "Fio"
num_nodes = 1
time_limit = "00:05:00"

  [Tests.cmd_args]
  fio_binary = "fio"

    [Tests.cmd_args.args]
    name = "cloudai-quickstart"
    filename = "/tmp/cloudai-fio-quickstart"
    rw = "randwrite"
    bs = "128k"
    size = "80m"
    iodepth = 1
    numjobs = 1
    group_reporting = true
    thread = true
```

</details>

**Slurm:** use the Slurm setup below. The example uses the public `openeuler/fio` image.
Change the host side of `extra_container_mounts` from `/tmp` to the storage directory you want to measure.
The container writes `/data/cloudai-fio-quickstart` inside that directory.

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/fio-slurm.toml
```

<details>
<summary>View configuration</summary>

**Scenario:** [fio-slurm.toml](conf/experimental/quickstart/fio-slurm.toml)

```toml
name = "quickstart-fio-slurm"

[[Tests]]
id = "fio"
name = "fio"
description = "Small file I/O benchmark"
test_template_name = "Fio"
num_nodes = 1
time_limit = "00:05:00"
extra_container_mounts = ["/tmp:/data"]

  [Tests.cmd_args]
  fio_binary = "fio"
  docker_image_url = "openeuler/fio:3.42-oe2403sp3"

    [Tests.cmd_args.args]
    name = "cloudai-quickstart"
    filename = "/data/cloudai-fio-quickstart"
    rw = "randwrite"
    bs = "128k"
    size = "80m"
    iodepth = 1
    numjobs = 1
    group_reporting = true
    thread = true
```

</details>

</details>

<details>
<summary>Training — Megatron Bridge, two eight-GPU H100 nodes</summary>

Run ten Qwen3-30B-A3B training steps. Use the Slurm setup below with eight GPUs per node,
and set `HF_TOKEN` in your environment before execution.

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/mbridge.toml
```

<details>
<summary>View configuration</summary>

**Scenario:** [mbridge.toml](conf/experimental/quickstart/mbridge.toml)

```toml
name = "quickstart-mbridge"

[[Tests]]
id = "training"
name = "training"
description = "Two-node H100 Qwen training benchmark"
test_template_name = "MegatronBridge"
num_nodes = 2
time_limit = "00:30:00"

  [[Tests.git_repos]]
  url = "https://github.com/NVIDIA-NeMo/Megatron-Bridge.git"
  commit = "r0.4.0"
  mount_as = "/opt/Megatron-Bridge"
  init_submodules = true

  [Tests.cmd_args]
  gpu_type = "h100"
  gpus_per_node = 8
  num_gpus = 16
  container_image = "nvcr.io#nvidia/nemo:26.02.00"
  model_family_name = "qwen"
  model_recipe_name = "qwen3_30b_a3b"
  domain = "llm"
  task = "pretrain"
  compute_dtype = "fp8_cs"
  max_steps = 10
  nemorun_version = "v0.9.0"

  [Tests.extra_env_vars]
  PYTHONPATH = "/opt/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH}"
```

</details>

</details>

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.4 or newer and clone CloudAI:

```bash
git clone https://github.com/NVIDIA/cloudai.git
cd cloudai
```

<details>
<summary>Standalone setup — local Fio</summary>

Install the Fio executable. On Ubuntu:

```bash
sudo apt-get install fio
cp conf/common/system/standalone_system.toml standalone.toml
```

The system configuration is:

```toml
name = "standalone_system"
scheduler = "standalone"
install_path = "./install"
output_path = "./results"
```

Run the local Storage command above from the repository root.

</details>

<details>
<summary>Slurm setup — inference, networking, storage, and training</summary>

Run CloudAI from a Slurm submit host with:

- Slurm CLI tools and access to the required partition and account.
- Container support for `srun` (such as Pyxis/Enroot) and access to the example's public image.
- Install, output, model, and input paths accessible on the execution nodes.

Save the following as `slurm.toml` in the repository root:

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

Replace both `YOUR_PARTITION` values and set GPU/task counts for your nodes. Set `account` at the top level if your
cluster requires one. For a CPU-only Fio allocation, omit `gpus_per_node` and set `ntasks_per_node = 1`.
Adjust `mpi` to the MPI plugin available on your cluster. Use shared absolute paths for `install_path` and
`output_path` if the repository directory is not visible on compute nodes.

Run one of the Slurm commands above. CloudAI installs declared workload prerequisites during execution;
model access and cluster container support must already be configured.

</details>

## Guide the next experiment

Open the results directory printed by CloudAI:

- Check `experiment.json` for execution status, metrics, and search results.
- Inspect the generated HTML reports to compare measurements.
- Read the execution's `stdout.txt` and `stderr.txt` when a workload fails or metrics are missing.

<details>
<summary>Search configurations — compare Ring and Tree with NCCL</summary>

After running the networking example, use the same Slurm setup to compare two algorithm choices:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/nccl-search.toml
```

<details>
<summary>View configuration</summary>

**Scenario:** [nccl-search.toml](conf/experimental/quickstart/nccl-search.toml)

```toml
name = "quickstart-nccl-search"

[[Tests]]
id = "all-reduce"
name = "all-reduce"
description = "Two-node NCCL all-reduce benchmark"
test_template_name = "NcclTest"
num_nodes = 2
time_limit = "00:05:00"
agent = "grid_search"
agent_metrics = ["default"]
agent_reward_function = "inverse"

  [Tests.cmd_args]
  docker_image_url = "nvcr.io#nvidia/pytorch:25.12-py3"
  subtest_name = "all_reduce_perf_mpi"
  ngpus = 1
  minbytes = "8M"
  maxbytes = "128M"
  iters = 20
  warmup_iters = 5
  stepfactor = 2

  [Tests.extra_env_vars]
  NCCL_ALGO = ["Ring", "Tree"]
```

</details>

Built-in `grid_search` evaluates both combinations. The objective uses inverse mean in-place latency;
the selected result is the successful step with the highest valid reward. Inspect `tests[].dse` and
`tests[].metrics` in `experiment.json`, alongside the DSE report and selected configuration.

Each additional search dimension multiplies the candidate count. Custom optimization agents can implement
feedback-driven selection and stopping policies for a compute budget.

</details>

<details>
<summary>Automate an experiment with Python</summary>

After the Standalone setup above, save this as `experiment.py`:

```python
from pathlib import Path

import cloudai.api

system = Path("standalone.toml")
scenario = Path("conf/experimental/quickstart/fio.toml")
cloudai.api.validate_scenario(scenario, system)
experiment = cloudai.api.run_experiment(scenario, system)
print(experiment.status, experiment.path)
```

Run `uv run python experiment.py`. Execution waits for completion and returns the data saved in `experiment.json`.
Workload failures appear in its status; configuration and setup errors raise exceptions.

</details>

Add workloads, systems and runners, reporters, or agents through CloudAI's public interfaces and registry.
For deeper customization, see [extension guides](https://nvidia.github.io/cloudai/extend/index.html),
[optimization](https://nvidia.github.io/cloudai/optimization/index.html), and the
[Python API reference](https://nvidia.github.io/cloudai/api.html).

## Contribute

See the [contribution guidelines](CONTRIBUTING.md) to report issues, propose changes, or add support for a workload.

## License

CloudAI is licensed under [Apache 2.0](LICENSE.md).

## External Materials

See the [Notice and Disclaimer](https://nvidia.github.io/cloudai/#external-materials-notice) in the CloudAI documentation before using external materials retrieved by CloudAI.
