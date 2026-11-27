---
name: cloudai-pr-review
description: Review a code diff or assess a PR comment for correctness, compatibility, design fit, and test coverage.
---

# Review a change

- For a full review, compare the head with its base. For a follow-up, start with the changes since the last review and any open findings. Widen the review if a shared contract changed.
- Use the contribution skill for the area under review when its description matches the change.
- Report a bug when the diff introduces or exposes a reachable failure. Trace a caller and input through the code, or give a reproducer. A user-authored config allowed by the public interface is valid evidence even if it has no example in the repo.
- Check who owns the state and how it reaches the caller. If new types or layers duplicate a responsibility, show the simpler path and the behavior it preserves.
- Check whether the tests exercise the changed contract. Identify an input that would fail if the implementation were wrong. Distinguish evidence from a generated command, a local test, and a live run.
- Give the location, trigger, and consequence. Omit purely hypothetical scenarios, unrelated pre-existing issues, and style preferences already settled by tooling. Check bot comments by the same standard. If there are no actionable findings, say so.
