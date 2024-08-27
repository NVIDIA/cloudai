# Development
This document targets developers who want to contribute to the project's core.


```mermaid
graph TD
    subgraph _core
        base_modules
        core_implementations
        registry
    end

    subgraph runners
        SlurmRunner
        StandaloneRunner
    end

    subgraph installers
        SlurmInstaller
        StandaloneInstaller
    end

    subgraph systems
        SlurmSystem
        StandaloneSystem
    end

    installers --> _core
    runners --> _core
    systems --> _core
```

## Core Modules
We use [import-linter](https://github.com/seddonym/import-linter) to ensure no core modules import higher level modules.

`Registry` object is a singleton that holds implementation mappings. Users can register their own implementations to the registry or replace the default implementations.

## Runners
TBD

## Installers
TBD

## Systems
TBD



```mermaid
---
title: Cloud AI
---
classDiagram
    TestRun *-- Test
    Job *-- TestRun
    Job *-- System

    class System {
        +run_cmd()
        +get_dir()
        +get_file()
        +...()
    }
    class Test {
        cmd_args
    }
    class TestRun {
        Test test
        TestRun[] dependencies
        int num_nodes
        str[] nodes
        str time_limit
    }
    class Job {
        int job_id
        tr: TestRun
        System system

        +run()
        +kill()

        +output_path()
        +job_id()
        +is_running()
        +is_done()
        +status()
    }
```
1. `Test` is a `TestDefinition` from Pydantic intro PR. It is a test with all arguments. Basically, it is a reflection of a Test.toml, where all params are defined or default values are used.
1. `TestRun` is a `Test` instance with `System`-specific parameters, like `num_nodes` for Slurm system.
1. `Job` is a single runnable unit. `Job` knows how to interact with the system to get required information like job status. It can consist of a single `TestRun` or multiple `TestRun`s. For Slurm system this means that a single sbatch script can contain one or multiple tests.

Notes:
1. `BaseRunner` and derivatives to be merged into Job class.