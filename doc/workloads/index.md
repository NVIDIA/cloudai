# Workloads Documentation

This section contains automatically generated documentation for all CloudAI workloads. Each workload provides specific functionality for running different types of tests and benchmarks.

## Available Workloads

```{toctree}
:maxdepth: 1
:caption: Workloads:

ai_dynamo
bash_cmd
chakra_replay
nccl
ddlb
nemo_run
nixl_bench
nixl_kvbench
nixl_perftest
sleep
slurm_container
ucc
```

## Adding New Workloads

To add documentation for a new workload:

1. **Add docstrings** to your Python classes and methods
1. **Create a markdown file** in `doc/workloads/` (e.g., `my_workload.md`)
1. **Add it to the toctree** in this index file

The documentation will be automatically generated during the build process!
