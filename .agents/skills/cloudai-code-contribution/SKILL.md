---
name: cloudai-code-contribution
description: Implement a feature, bug fix, or refactor affecting CloudAI behavior or interfaces.
---

# Code contribution

- Follow the repository’s [contribution guide](../../../CONTRIBUTING.md) for Python style. Before handing off, check the diff for unrelated formatting, renames, or cleanup.
- Trace the affected interface (Python API, TOML, generated command, or result file). Include callers who supply their own configs or extensions. Identify who owns the state.
- When starting a feature or changing its expected behavior, briefly state how you’ll verify it. On follow-up edits, reuse the existing plan and call out only what changes. For a bug, reproduce the failure when practical.
- Reuse the existing representation and extension point where they fit. Introduce another type when it has a distinct owner, lifetime, or invariant. Keep standard interfaces usable by their current callers.
- Extend the closest existing test or fixture. Add a case when it catches a distinct regression. Exercise the normal entry point and report what ran, what it showed, and any claim that still needs a live workload.
