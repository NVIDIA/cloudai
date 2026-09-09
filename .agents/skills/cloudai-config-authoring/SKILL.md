---
name: cloudai-config-authoring
description: Create, adapt, review, or troubleshoot CloudAI system, test, and test-scenario TOML configurations. Use for CloudAI config-file requests, not for implementing workload Python code.
---

# CloudAI config authoring

## Sources and placement

- Adapt the closest existing config and place new examples in the matching common, experimental, or release subtree.
- Confirm fields against the registered system class in `src/cloudai/systems/` or the workload's `TestDefinition` and
  `CmdArgs` models. Check template names in `src/cloudai/registration.py`.
- For scenario fields and overrides, use `src/cloudai/models/scenario.py` and `src/cloudai/test_scenario_parser.py`.
  Current models, validators, and parsers define the schema.

## Configuration semantics

- Preserve values unrelated to the requested change, including names, image versions, arguments, and references.
- Keep test `name` values unique within their test directory; use registered `test_template_name` values exactly.
- Each scenario test uses one definition mechanism: `test_name`, `path`, or inline `test_template_name`. A `path`
  resolves relative to the scenario file's directory. Inline definitions also require `name` and `description`.
  Preserve the existing mechanism when editing; follow comparable scenarios when creating.
- Preserve list-versus-scalar semantics: lists can enable design-space exploration or be intrinsically list-valued.
  Check the model, validators, and `dse_excluded_args` before changing the form.
- Use illustrative environment values in reusable examples; keep user-specific paths and machine assignments in
  explicitly environment-specific configs.

## Validation

- Path-based and inline-only scenarios do not require `--tests-dir` for validation.
- Check formatting with `uv run --locked --extra dev taplo fmt --check <changed-tomls>`.
