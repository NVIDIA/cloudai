<h1 align="center">CloudAI</h1>

<p align="center"><strong>The execution framework for AI-driven research.</strong></p>

<p align="center">
  <a href="LICENSE.md"><img src="https://img.shields.io/badge/License-Apache_2.0-76B900" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.10%2B-76B900" alt="Python 3.10 or newer"></a>
</p>

<p align="center">
  <a href="#supported-workloads">Workloads</a> &nbsp;·&nbsp;
  <a href="#run-your-first-experiment">Get started</a> &nbsp;·&nbsp;
  <a href="#optimize-configurations">Optimization</a> &nbsp;·&nbsp;
  <a href="#automate-and-extend">Python &amp; extensions</a> &nbsp;·&nbsp;
  <a href="#contribute">Contribute</a>
</p>

Run the workloads your task requires, collect evidence, and use the results to guide the next experiment.
Combine experiments into a broader investigation using existing workloads or your own.

| Run workloads | Find better configurations | Add what your task needs |
| :--- | :--- | :--- |
| One CLI, Python API, and configuration model across workloads. | Grid search or optimization agents that guide experiments within a compute budget. | Public interfaces for workloads, execution systems, reporters, and agents. |

![CloudAI connects experiment configuration to workload execution, metrics, and reusable results. Custom agents can use measured feedback. Public interfaces support workloads, systems and runners, reporters, and agents.](doc/_static/cloudai-experiments.svg)

## Supported workloads

| Area | Integrations |
| :--- | :--- |
| **Inference** | Dynamo, vLLM, SGLang, SRT-Slurm, Triton Inference |
| **Training** | Megatron Bridge, MegatronRun, NeMo Run, DDLB |
| **Networking & communication** | NCCL, UCC, OSU, DeepEP, MoE Benchmark, NIXL Bench, NIXL EP, NIXL KVBench, NIXL Perftest |
| **Storage** | Fio; storage-oriented transfers through NIXL Bench |
| **Simulation & prediction** | AIConfigurator, Dynamo Mocker, Chakra Replay |
| **Custom execution** | Bash commands, Slurm containers, Sleep for scheduling checks |
| **Legacy training** | NeMo Launcher, JAX Toolbox (GPT, Grok, Nemotron) |

<details>
<summary><strong>Full workload × system compatibility</strong></summary>

Names below are the exact `test_template_name` values. A check mark indicates a registered execution strategy;
each workload still requires its own software and resources.

<!-- Keep this matrix aligned with doc/reference/workload-support.rst and src/cloudai/registration.py. -->

| Workload | Slurm | Standalone | Kubernetes | RunAI | LSF |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Aiconfigurator` | — | ✅ | — | — | — |
| `AIDynamo` | ✅ | — | ✅ | — | — |
| `BashCmd` | ✅ | — | — | — | — |
| `ChakraReplay` | ✅ | — | — | — | — |
| `DDLBTest` | ✅ | — | — | — | — |
| `DeepEP` | ✅ | — | — | — | — |
| `DynamoMocker` | — | ✅ | — | — | — |
| `Fio` | ✅ | ✅ | — | — | — |
| `JaxToolboxGPT` | ✅ | — | — | — | — |
| `JaxToolboxGrok` | ✅ | — | — | — | — |
| `JaxToolboxNemotron` | ✅ | — | — | — | — |
| `MegatronBridge` | ✅ | — | — | — | — |
| `MegatronRun` | ✅ | — | — | — | — |
| `MoEBenchmark` | ✅ | — | — | — | — |
| `NcclTest` | ✅ | — | ✅ | ✅ | — |
| `NeMoLauncher` | ✅ | — | — | — | — |
| `NeMoRun` | ✅ | — | — | — | — |
| `NIXLBench` | ✅ | — | — | — | — |
| `NixlEP` | ✅ | — | — | — | — |
| `NIXLKVBench` | ✅ | — | — | — | — |
| `NixlPerftest` | ✅ | — | — | — | — |
| `OSUBench` | ✅ | — | — | — | — |
| `sglang` | ✅ | — | — | — | — |
| `Sleep` | ✅ | ✅ | ✅ | — | ✅ |
| `SlurmContainer` | ✅ | — | — | — | — |
| `SRTSlurm` | ✅ | — | — | — | — |
| `TritonInference` | ✅ | — | — | — | — |
| `UCCTest` | ✅ | — | — | — | — |
| `vllm` | ✅ | — | — | — | — |

</details>

## Run your first experiment

### 1. Get CloudAI

Install `uv` 0.13.4 or newer, then clone CloudAI:

```bash
git clone https://github.com/NVIDIA/cloudai.git
cd cloudai
```

### 2. Choose an experiment

Fio runs locally without a GPU. The vLLM, NCCL, and Megatron Bridge examples use the shared Slurm setup below.

<details>
<summary><strong>Storage — run Fio locally</strong></summary>

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
<summary><strong>Slurm setup — required for the GPU examples</strong></summary>

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
<summary><strong>Inference — vLLM on one GPU</strong></summary>

After the Slurm setup above, benchmark Qwen3-0.6B on one GPU:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/vllm.toml
```

The [scenario](conf/experimental/quickstart/vllm.toml) sends 30 prompts at concurrency 16,
with 16 input and 128 output tokens. Change the model, traffic, or parallelism settings to match your task.

</details>

<details>
<summary><strong>Networking — NCCL across two nodes</strong></summary>

After the Slurm setup above, measure all-reduce across two nodes:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/nccl.toml
```

The [scenario](conf/experimental/quickstart/nccl.toml) measures message sizes from 8 MiB to 128 MiB.
Change the collective, message sizes, or node count to investigate another communication pattern.

</details>

<details>
<summary><strong>Training — Megatron Bridge on two eight-GPU H100 nodes</strong></summary>

Complete the Slurm setup above and set `HF_TOKEN` in your environment. Run ten Qwen3-30B-A3B training steps:

```bash
uv run cloudai run --system-config slurm.toml \
    --test-scenario conf/experimental/quickstart/mbridge.toml
```

The [scenario](conf/experimental/quickstart/mbridge.toml) specifies the model recipe, GPU layout,
container, and training duration. Start with a short run before increasing the duration or scale.

</details>

## Inspect results

Open the results directory printed by CloudAI:

| Artifact | Use it to |
| :--- | :--- |
| `experiment.json` | Read execution status, metrics, and selected search results. |
| HTML reports | Inspect measurements and compare configurations. |
| `stdout.txt` / `stderr.txt` | Diagnose failures or missing metrics. |

Keep model, input, resource, and measurement conditions consistent when comparing configurations.

## Optimize configurations

Evaluate a grid of choices, or connect an agent that selects the next experiment from measured feedback.
Custom agents control stopping according to their compute-budget policy.

<details>
<summary><strong>Search configurations — compare two NCCL algorithms</strong></summary>

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

## Automate and extend

Use the Python API to run experiments from your own tools or an AI agent. Add workloads, execution systems,
reporters, and optimization agents through public interfaces and the registry.

<details>
<summary><strong>Automate experiments with Python</strong></summary>

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

<details>
<summary><strong>External materials — notice and disclaimer</strong></summary>

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.
Those retrieved materials are not distributed with this software and are governed solely by separate terms,
conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable
terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved
materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The
author makes no representations or warranties regarding any retrieved materials, and assumes no liability for
any losses, damages, liabilities or legal consequences from your use or inability to use this software or any
retrieved materials. Use this software and the retrieved materials at your own risk.

</details>
