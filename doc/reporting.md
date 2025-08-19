# Reporting
This document describes the reporting system in CloudAI.


## Overview
CloudAI has two reporting levels: per-test (per each case in a test scenario) and per-scenario (per each test scenario). All reports are generated after the test scenario is completed as part of main CloudAI process. For Slurm this means that login node is used to generate reports.

Per-test reports are linked to a particular workload type (e.g. `NcclTest`). All per-test reports are implemented as part of `per_test` scenario report and can be enabled/disabled via single configuration option, see [Enable, disable and configure reports](#enable-disable-and-configure-reports) section.

To list all available reports, one can use `cloudai list-reports` command. Use verbose output to also print report configurations.


## Notes and general flow
1. All reports should be registered via `Registry()` (`.add_report()` or `.add_scenario_report()`).
1. Scenario reports are configurable via system config (Slurm-only for now) and scenario config.
1. Configuration in a scenario config has the highest priority. Next, system config is checked. Then it defaults to report config from the registry.
1. Then report is generated (or not) according to this final config.


## Enable, disable and configure reports
**NOTE** Only scenario-level reports can be configured today.

To enable or disable a report, one needs to do it via System configuration:
```toml
[reports]
per_test = { enable = false }
status = { enable = true }
```


## Report registration
Report registration is done via `Registry` class:

```python
Registry().add_scenario_report("per_test", PerTestReporter, ReportConfig(enable=True))
```


## Report configuration implementation
Each report can define its own configuration which is constructed and passed as an argument to `Registry.add_scenario_report` method. `reports` field is parsed during TOMLs reading and respective Pydantic model is created.

For example, we can define a custom report configuration:
```python
class CustomReportConfig(ReportConfig):
    greeting: str
```

```python
Registry().add_scenario_report("custom", CustomReport, CustomReportConfig(greeting="default value"))
```

And use it in a test scenario:
```toml
[reports]
custom = { enable = true, greeting = "Hello, world!" }
```
