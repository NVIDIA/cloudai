# CloudAI repository guidance

## Implementation conventions

- Avoid over-engineering
- Prefer Google Python style guide:
  - Blend into existing code
  - No asserts in production code
  - Prefer absolute imports (`import x`).
    Use `import y from x` when `x.y` is too long.
    Use relative imports when existing code uses it.

- Backwards compatibility is very important. CloudAI may be integrated into other tools. Users maintain their own
  CloudAI configs that we may never see. Backwards incompatible changes should be avoided unless explicitly asked. In
  that case the changes must be highlighted.
- Update affected documentation in `README.md` or `doc/` when public behavior changes.
- Import public core APIs through `cloudai.core`, respect import-linter boundaries, and use existing lazy-import
  mechanisms for heavy modules. Follow established workload structure and registration patterns.
- Follow `CONTRIBUTING.md`, including SPDX headers and mirrored tests for new Python modules.

## Verification

- Start with focused tests: `uv run --locked --extra dev pytest <test-paths>`.
- Run `uv run --locked --extra dev pre-commit run --files <changed-files>` and review formatter edits.

## Contribution

- Do not commit, push, open or modify pull requests, or run remote jobs unless explicitly asked to do so.
- Create PRs as drafts (`gh pr create --draft`). Leave marking PRs ready for review to humans unless asked to do so.
- Follow PR requirements in `CONTRIBUTING.md` and use `.github/PULL_REQUEST_TEMPLATE.md`.
- Keep public configs reusable (no internal resources). Never commit credentials, private artifacts, or internal
  hostnames.
