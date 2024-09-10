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
        Path _base_output_dir
        int num_nodes
        str[] nodes
        str time_limit

        +output_dir()
    }
    class Job {
        int job_id
        TestRun tr
        System system

        +run()
        +kill()

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

# Execution flow through the system
```mermaid
flowchart TB
    ur(A user runs cloudai specifying CLI arguments)
    ur --> parsing
    sp --> ttp
    ttp --> tp
    tp --> tsp

    subgraph parsing
    sp(First, System TOML is parsed into System object. It contains all system-specific parameters, plus output and installation directories.)
    ttp(Then, all Test Template TOMLs are parsed into TestTemplate objects. These objects contain env and cmd arguments. TestTemplate requires a System object to be construct Strategies: Install, CommandGen, etc. *)
    tp(Then, all Test TOMLs are parsed into Test objects. These objects contain TestTemplate as a property. And again, use env and cmd arguments, but also have extra_env and extra_cmd arguments.)
    tsp(Finally, Test Scenario TOML is parsed into TestScenario object. It constructs TestRun objects, which contain Test objects and some run-specific parameters like num_nodes.)
    end

    parsing --> execution    
    subgraph execution
    end

    execution --> report_generation
    subgraph report_generation
    end
```
\* Some Strategies are inherited from TestTemplateStrategy and require System, env, and cmd arguments for their construction. Other Strategies fo not require any arguments.

## output directory
Output directory is set per Cloud AI invocation. It is constructed as follows:
1. `BaseOutputDir = System.output_directory + TestScenario.name + CurrentTime`
1. Each test then adds its own subdirectory to the output directory like `BaseOutputDir/TestName`

