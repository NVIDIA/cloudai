---
name: cloudai-config-authoring
description: Create, adapt, review, or troubleshoot CloudAI system, test, and test-scenario TOML configurations. Use for CloudAI config-file requests, not for implementing workload Python code.
---

# CloudAI config authoring

Produce the smallest configuration change that meets the user's intent and remains compatible with existing CloudAI
behavior.

## Establish the configuration shape

- Determine whether the task needs a system config, test config, test-scenario config, or a coordinated set of them.
- Establish the scheduler, workload template, intended placement, and any environment-specific inputs. Infer from the
  nearest relevant configs when the choice is clear; ask only when a missing choice would materially change the result.
- Inspect the closest existing config before authoring. Prefer adapting a working example over creating a new layout or
  naming scheme.

## Use live sources of truth

- For system fields, inspect the registered system class under `src/cloudai/systems/` and comparable files under
  `conf/**/system/`.
- For test fields, confirm the `test_template_name` in `src/cloudai/registration.py`, then inspect that workload's
  `TestDefinition` and `CmdArgs` models and comparable files under `conf/**/test/`.
- For scenario fields and merge behavior, inspect `src/cloudai/models/scenario.py`,
  `src/cloudai/test_scenario_parser.py`, and comparable files under `conf/**/test_scenario/`.
- Use the models, validators, registry, and parser behavior as the schema. Do not create a parallel schema or assume that
  a documented example contains every supported field.

## Author safely

- Preserve existing names, defaults, image versions, command arguments, and cross-file references unless the user asks
  to change them. Do not perform unrelated version refreshes while preparing a config.
- Keep every test `name` unique in its test directory and use a registered `test_template_name` exactly as spelled.
- A scenario test must select its definition through `test_name`, relative `path`, or an inline `test_template_name`.
  Preserve the existing mechanism when editing; when creating, choose the mechanism used by the closest comparable
  scenario.
- Do not turn a scalar into a list merely to offer alternatives. Lists commonly enable design-space exploration, while
  some workload fields are intrinsically list-valued. Confirm the model, validators, `dse_excluded_args`, and nearby
  configs before choosing.
- Place broadly reusable examples under the established `conf/common/` structure. Follow an existing release or
  experimental subtree when the config is specific to that suite or workload; do not invent a new top-level hierarchy.
- Keep reusable configs free of credentials, private artifacts, internal NVIDIA hostnames, personal paths, and concrete
  machine assignments. Use user-provided environment values only in explicitly environment-specific files.

## Validate and report

- Validate a system or test config with `uv run --locked cloudai verify-configs <config-path>`.
- Validate a scenario with
  `uv run --locked cloudai verify-configs --tests-dir <matching-test-directory> <scenario-path>` when it references
  tests by `test_name`. For a coordinated config tree, validate the smallest containing directory with the exact
  `--tests-dir` expected at runtime.
- Run `uv run --locked --extra dev taplo fmt --check <changed-tomls>` when Taplo is available through the development
  environment. Fix only the target files.
- Use `cloudai dry-run` only when a suitable system config and all required local inputs are available. A dry run does
  not replace scheduler or hardware validation.
- Report the files created or changed, the validation commands and outcomes, and any remote, scheduler, accelerator, or
  hardware checks that were not performed.
