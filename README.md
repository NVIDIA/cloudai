# CloudAI

**The execution framework for AI-driven research.**

Run the workloads your task requires, collect evidence, and use the results to guide the next experiment.
Combine experiments into a broader investigation using existing workloads or your own.

![CloudAI connects experiment configuration to workload execution, metrics, and reusable results. Custom agents can use measured feedback. Public interfaces support workloads, systems and runners, reporters, and agents.](doc/_static/cloudai-experiments.svg)

[Extend CloudAI](https://nvidia.github.io/cloudai/DEV.html) ·
[Configure optimization agents](https://nvidia.github.io/cloudai/USER_GUIDE.html#agents-configuration)

## Run your first experiment

Start with an example below, or explore [all supported workloads](https://nvidia.github.io/cloudai/workloads/index.html).

| Experiment | Start here |
| --- | --- |
| Inference | [vLLM: Qwen3-0.6B on one GPU](https://nvidia.github.io/cloudai/?benchmark=inference#choose-a-workflow) |
| Networking | [NCCL: all-reduce on two nodes](https://nvidia.github.io/cloudai/?benchmark=networking#choose-a-workflow) |
| Storage | [Fio: file I/O on standalone or Slurm](https://nvidia.github.io/cloudai/?benchmark=storage#choose-a-workflow) |
| Training | [Megatron Bridge: Qwen3-30B-A3B on two eight-GPU H100 nodes](https://nvidia.github.io/cloudai/?benchmark=training#choose-a-workflow) |

## Get started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.4 or newer and clone CloudAI:

```bash
git clone https://github.com/NVIDIA/cloudai.git
cd cloudai
```

Follow the [Standalone](https://nvidia.github.io/cloudai/#quickstart-standalone) or
[Slurm](https://nvidia.github.io/cloudai/#quickstart-slurm) setup, then run an example above.
See [installation and CLI usage](https://nvidia.github.io/cloudai/getting-started.html) for more options.

## Guide the next experiment

Open the results directory printed by CloudAI. Use [HTML reports and `experiment.json`](https://nvidia.github.io/cloudai/reporting.html)
to inspect measurements, or consume results through the [Python API](https://nvidia.github.io/cloudai/api.html).

Use built-in `grid_search` to evaluate every configured combination, or connect your own
[optimization agent](https://nvidia.github.io/cloudai/USER_GUIDE.html#agents-configuration) to select configurations from
measured feedback. Custom agents can implement adaptive selection and stopping policies to work within a compute budget.

Add [workloads, systems and runners, reporters, or agents](https://nvidia.github.io/cloudai/DEV.html)
through CloudAI's public interfaces and registry.

## Contribute

See the [contribution guidelines](CONTRIBUTING.md) to report issues, propose changes, or add support for a workload.

## License

CloudAI is licensed under [Apache 2.0](LICENSE.md).

## External Materials

See the [Notice and Disclaimer](https://nvidia.github.io/cloudai/#external-materials-notice) in the CloudAI documentation before using external materials retrieved by CloudAI.
