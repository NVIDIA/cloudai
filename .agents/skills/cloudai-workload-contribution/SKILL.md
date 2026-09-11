---
name: cloudai-workload-contribution
description: Create, modify, or review CloudAI workload implementations. Use for workload Python code, not for TOML-only configuration changes.
---

This skill is a set of guidelines when working on workloads implementation

## Implementation guidelines

- User configs is a trusted data. Don't over-validate test definitions
- The workload implementation is intended to be pass-through, which means that CloudAI:
  - defines benchmark shape (processes, installables, etc.)
  - translates TOML test config into workload interface submission so that user can use the workload fully
  - doesn't (re-)define underlying workload parameters unless required for workload submission and clean code

- CloudAI cannot support every possible cluster-specific hardware/software setup in terms of benchmark
  startup/finalization. Prefer generic solutions instead of specific technologies support in the workloads
  implementation. The generic solutions are:

    - container mounts
    - installables
    - pre/post-srun scripts (custom per workload; some workloads already support it)
    - pre/post-tests for heavy lifting hooks that need an srun

- Fetching workloads sources to understand how to use them. Use `results/vendor-src` folder for it. When
  working under a worktree, re-use this folder from the main repo checkout. Be aware of the checkout version. Have a
  single checkout for one code source (switch checkouts)
- Make the most of parent CloudAI command generation classes so that the workload supports all the builtin features,
  like single-sbatch, sbatch directives, pre/post-test hooks, DSE/CloudAIGym. When it's too complicated to support one
  - do not (80-20 rule)

## Testing guidelines

- Don't produce too many unit-tests covering a small feature. More tests != better
- Prefer maintaining end-to-end tests (tests/test_acceptance.py). However, it must be only one or two such tests per
  workload (they should be really different in shape)
- Existing workload TOML configs from `conf/` must stay supported. One may extend them with new features. One test case
  in a scenario supporting a new workload feature is enough
- If a workload may result in diverse execution shapes (number of processes and their orchestration), then prepare that
  many test cases (not scenarios)
