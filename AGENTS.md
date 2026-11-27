# CloudAI repository guidance

## Implementation conventions

- CloudAI is a public framework. Users write their own TOMLs and extensions. Files under `conf/` give examples.
  Use the public schema and extension contracts to assess compatibility. Preserve behavior unless a breaking change is requested,
  and document its impact.
- Prefer the simplest change that meets the request. Follow the [contributor guide](CONTRIBUTING.md#development-and-coding-standards) for Python style and PR scope.
- Update affected documentation in `README.md` or `doc/` when public behavior changes.
- Import public core APIs through `cloudai.core`, respect import-linter boundaries, and use existing lazy-import
  mechanisms for heavy modules. Follow established workload structure and registration patterns.
- Follow `CONTRIBUTING.md`, including SPDX headers and mirrored tests for new Python modules.

## Verification

- Start with focused tests: `uv run --locked --extra dev pytest <test-paths>`.
- Run `uv run --locked --extra dev pre-commit run --files <changed-files>` and review formatter edits.

## Contribution

- Use `cloudai-code-contribution` for features, fixes, and refactors.
- Use `cloudai-pr-review` for code reviews and review-comment triage.
- Do not commit, push, open or modify pull requests, or run remote jobs unless explicitly asked to do so.
- Follow PR requirements in `CONTRIBUTING.md` and use `.github/PULL_REQUEST_TEMPLATE.md`.
- Use `.agents/skills/pr-management/SKILL.md` for PR creation/updates and after every push to a branch with an open PR.
- Use public examples and resources in configs, docs, fixtures, PR text, and screenshots. Keep credentials,
  private artifacts, and internal hostnames out of contributions.
