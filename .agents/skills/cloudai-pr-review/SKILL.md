---
name: cloudai-pr-review
description: Review a code diff or assess a PR comment for correctness, compatibility, design fit, and test coverage.
---

# Review a change

- Read the request and compare the head with its base. Trace an affected caller using a concrete input, including an existing config or result when its format changes.
- Check who owns the state and how it reaches the caller. If new types or layers duplicate a responsibility, show the simpler path and the behavior it preserves.
- Check whether the tests exercise the changed contract. Identify an input that would fail if the implementation were wrong. Distinguish evidence from a generated command, a local test, and a live run.
- Lead each finding with its location, trigger, and observable consequence. Check bot suggestions against the actual diff and repository rules. If there are no actionable findings, say so.
