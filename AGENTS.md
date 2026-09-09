# CloudAI repository guidance

## Project context

- Many contributors use NVIDIA infrastructure, but ordinary development and config checks must work without internal
  access. Keep public examples reusable; never commit credentials, private artifacts, or internal hostnames.

## Compatibility first

- Preserve CLI commands, options, environment variables and exit behavior; TOML fields and defaults; workload template
  names; generated jobs; and result, metadata, and report formats.
- For supported workload behavior changes, identify affected configurations and tests and add regression coverage for
  existing inputs. Prefer opt-in additions and unchanged defaults. Explicitly requested breaking changes need migration
  or deprecation handling.

## Implementation conventions

- Follow supported Python versions and checks in `pyproject.toml`, `.pre-commit-config.yaml`, and
  `.github/workflows/ci.yml`.
- Use current implementation, registered classes, Pydantic models, and nearby tests as sources of truth. Update affected
  documentation in `README.md` or `doc/` when public behavior changes.
- Import public core APIs through `cloudai.core`, respect import-linter boundaries, and use existing lazy-import
  mechanisms for heavy modules. Follow established workload structure and registration patterns.
- Follow `CONTRIBUTING.md`, including SPDX headers and mirrored tests for new Python modules.
- Load relevant skills from the canonical `.agents/skills/` directory. Preserve unrelated user changes.

## Verification

- Start with focused tests: `uv run --locked --extra dev pytest <test-paths>`.
- Run `uv run --locked --extra dev pre-commit run --files <changed-files>` and review formatter edits.
- Hardware-dependent validation may require specific infrastructure. State its prerequisites, what was tested, and any
  remote, scheduler, or hardware validation left unverified.

## Git and external actions

- Do not commit, push, open or modify pull requests, or run remote jobs unless the user explicitly requests that action.
- Follow the DCO sign-off and PR requirements in `CONTRIBUTING.md` and use `.github/PULL_REQUEST_TEMPLATE.md`.
