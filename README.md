# CloudAI

**Benchmark and optimize AI systems.**

CloudAI is an open-source framework for benchmarking and optimizing AI systems. Run training, inference,
communication, and storage workloads through one experiment workflow. Use built-in grid search or connect custom
optimization agents to explore configurations and identify the best results observed for your objective.

[Get started](https://nvidia.github.io/cloudai/#getting-started) ·
[Choose a workflow](https://nvidia.github.io/cloudai/#choose-a-workflow) ·
[Workload compatibility](https://nvidia.github.io/cloudai/workloads/index.html)

## From benchmarks to better configurations

| Run different benchmarks | Search configurations | Compare and reuse results |
| --- | --- | --- |
| Use a common CLI, Python API, and configuration model for workloads such as NCCL, NeMo, vLLM, SGLang, and Fio. | Define configuration choices and a performance objective. Use grid search for small spaces or integrate your own search policy. | Inspect performance trade-offs, read structured experiment results, and reuse the best configuration discovered during a search. |

CloudAI's optimization agents select configurations and receive workload measurements as feedback. The built-in
`grid_search` agent evaluates every combination in the configured search space. Custom agents can implement adaptive
selection and stopping policies through the public agent interface.

**Plan around your compute budget:** start with a baseline, choose meaningful search dimensions, and estimate the
number and cost of evaluations before launching. Keep grid searches small; use a custom agent when you need an
adaptive search policy. See [configuration search](https://nvidia.github.io/cloudai/#configuration-search).

## Choose a workflow

| I want to… | Start here |
| --- | --- |
| Measure communication or storage performance | [NCCL](https://nvidia.github.io/cloudai/workloads/nccl.html), [NIXL](https://nvidia.github.io/cloudai/workloads/nixl_bench.html), or [Fio](https://nvidia.github.io/cloudai/workloads/fio.html) |
| Benchmark training configurations | [Megatron Bridge](https://nvidia.github.io/cloudai/workloads/megatron_bridge.html) or [NeMo Run](https://nvidia.github.io/cloudai/workloads/nemo_run.html) |
| Compare inference configurations | [vLLM](https://nvidia.github.io/cloudai/workloads/vllm.html), [SGLang](https://nvidia.github.io/cloudai/workloads/sglang.html), and [latency–throughput trade-offs](https://nvidia.github.io/cloudai/pareto-frontiers.html) |
| Explore a configuration space | [Configuration search](https://nvidia.github.io/cloudai/#configuration-search) and [agent configuration](https://nvidia.github.io/cloudai/USER_GUIDE.html#agents-configuration) |
| Automate experiments and consume their results | [Python API](https://nvidia.github.io/cloudai/api.html) and [experiment output](https://nvidia.github.io/cloudai/reporting.html#unified-experiment-output) |

Workload availability depends on the execution environment. Check the
[support matrix](https://nvidia.github.io/cloudai/workloads/index.html) before selecting a workload.

## Get started

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.13.4 or newer to manage Python and dependencies:

```bash
git clone https://github.com/NVIDIA/cloudai.git
cd cloudai
uv run cloudai --help
```

Choose a workload above, then configure [where it will run](https://nvidia.github.io/cloudai/systems.html) and prepare
its [prerequisites](https://nvidia.github.io/cloudai/workloads/workloads_requirements_installation.html).
CloudAI uses three configuration types: a **system** describes where to run, a **test definition** describes the
workload and its settings, and a **scenario** selects the test cases to execute.

Use [`dry-run`](https://nvidia.github.io/cloudai/#dry-run) to inspect generated workload commands before execution.
After an experiment, open its HTML reports or read `experiment.json` in the results directory. Configuration searches
also produce a DSE report and a selected test configuration; see
[report generation](https://nvidia.github.io/cloudai/#generate-report).

For installation alternatives and CLI examples, see the [documentation](https://nvidia.github.io/cloudai/#getting-started).

## Contribute

See the [contribution guidelines](CONTRIBUTING.md) to report issues, propose changes, or add support for a workload.

## License

CloudAI is licensed under [Apache 2.0](LICENSE.md).
