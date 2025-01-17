---
id: cli
---

# CLI

Cloud AI provides a command line interface (CLI) to interact with workloads.


## `cloudai run`
Runs a specified workload on a particular system. Might require [`cloudai install`](#cloudai-install) to be run first.

```bash
usage: cloudai run [-h] \
    --system-config SYSTEM_CONFIG \
    --tests-dir TESTS_DIR \
    --test-scenario TEST_SCENARIO \
    [--output-dir OUTPUT_DIR]
```

| Option | Required | Description |
|:-------|:---------|:------------|
| `--system-config` | yes | System TOML config |
| `--tests-dir` | yes | Path to Test TOML files, should be a valid directory |
| `--test-scenario` | yes | Test Scenario TOML config |
| `--output-dir` | no | Override default `output-dir` specified in System TOML config |


## `cloudai dry-run`
Same as [`cloudai run`](#cloudai-run), but does not actually run the workload. Useful for debugging and configuration validation.


## `cloudai install`
TBD


## `cloudai uninstall`
TBD


## `cloudai generate-reports`
TBD


## `cloudai verify-configs`
TBD
