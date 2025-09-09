# Workloads Documentation

This section contains automatically generated documentation for all CloudAI workloads. Each workload provides specific functionality for running different types of tests and benchmarks.

## Available Workloads

```{toctree}
:maxdepth: 1
:caption: Workloads:

bash_cmd
nccl
```

## Adding New Workloads

To add documentation for a new workload:

1. **Add docstrings** to your Python classes and methods
1. **Create a markdown file** in `doc/workloads/` (e.g., `my_workload.md`)
1. **Add it to the toctree** in this index file

The documentation will be automatically generated during the build process!
