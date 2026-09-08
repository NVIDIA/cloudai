# CloudAI repository guidance

## Project context

- CloudAI is an open-source, CLI-only benchmark runner for data-center AI systems. Do not introduce a service, web API,
  or GUI unless the user explicitly asks for one.
- This repository is public even though many contributors use NVIDIA infrastructure. Keep changes, examples, and
  validation reproducible without internal access. Never commit credentials, private artifacts, internal hostnames, or
  environment-specific secrets.
- Follow the user's explicit instructions when they conflict with general guidance in this file or a repository skill.

## Sources of truth

- Read the relevant implementation and nearby tests before changing behavior. Use registered classes and Pydantic
  models as the source of truth for supported systems, workloads, and configuration fields.
- Treat `pyproject.toml`, `.pre-commit-config.yaml`, and `.github/workflows/ci.yml` as authoritative for supported Python
  versions and automated checks. Do not duplicate dependency or tool versions in guidance.
- Use `README.md`, `CONTRIBUTING.md`, and `doc/` for documented behavior, but reconcile them with the current code and
  update them when user-visible behavior changes.
- Repository skills live in `.agents/skills/`. Load the matching `SKILL.md` when a task fits its description. The
  `.claude/skills` path is an adapter to the same canonical directory; do not maintain a second copy there.

## Compatibility first

- Treat public CLI commands, options, environment variables, exit behavior, TOML fields and defaults, workload template
  names, generated job behavior, result layout, metadata, and report formats as compatibility surfaces.
- For supported workloads, keep existing committed configurations working. Before changing one, inspect its common,
  release, and experimental configurations, parser and registration entries, documentation, and focused tests.
- Prefer additive, optional configuration with defaults that preserve current behavior. Do not remove or rename a
  compatibility surface, tighten accepted input, or change a default unless the task explicitly requires it and the
  change includes migration or deprecation handling plus regression coverage.
- Preserve list-versus-scalar configuration semantics. Lists often define a design-space search, but some fields are
  intrinsically list-valued; confirm the model and existing configs before changing either form.

## Implementation conventions

- Support Python 3.10 through 3.14. Use the project's `uv` environment and follow the configured Ruff, Pyright,
  import-linter, Vulture, and Taplo rules.
- Respect the package boundaries enforced by import-linter. Import public core APIs through `cloudai.core`; do not reach
  into `cloudai._core` from higher-level packages.
- Keep heavy optional modules behind the existing lazy-import mechanisms. The configured banned module-level imports
  include pandas, NumPy, Kubernetes, and Bokeh.
- Follow established workload structure and registration patterns. A workload change may require coordinated updates to
  its definition, scheduler strategy, public exports, central registration, report or grading logic, tests,
  documentation, and sample configs.
- Add the repository SPDX copyright and Apache-2.0 headers to new Python files. Add corresponding focused tests for new
  Python source modules as required by `CONTRIBUTING.md`.
- Keep diffs scoped. Preserve unrelated user changes and do not reformat or modernize adjacent code without a reason
  connected to the task.

## Verification

- Start with focused tests: `uv run --locked --extra dev pytest <test-paths>`.
- For TOML changes, run `uv run --locked cloudai verify-configs <config-path>`. When a scenario uses `test_name`, pass
  the exact matching test directory with `--tests-dir`.
- For workload behavior changes, add or update focused unit tests and exercise a representative `cloudai dry-run` when
  the required system config and prerequisites are available.
- Before handoff, run the relevant pre-commit hooks, normally
  `uv run --locked --extra dev pre-commit run --files <changed-files>`, and review any formatter edits. Broaden to the
  full CI commands when the risk or change scope warrants it.
- Do not claim remote, scheduler, accelerator, or hardware validation that was not performed. State what ran, what
  passed, and what remains unverified.

## Git and external actions

- Do not commit, push, open or modify pull requests, or run remote jobs unless the user explicitly requests that action.
- When asked to commit, follow the DCO sign-off requirement in `CONTRIBUTING.md`. Keep commit and pull-request summaries
  specific, including compatibility impact and exact validation performed.
- Keep procedural CI triage, pull-request preparation, and remote-lab execution in task-specific skills rather than
  expanding this always-loaded file.
