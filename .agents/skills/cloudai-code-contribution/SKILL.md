---
name: cloudai-code-contribution
description: Implement a feature, bug fix, or refactor affecting CloudAI behavior or interfaces.
---

# Code contribution

- Trace one real caller through the existing implementation. Identify who owns the state and which interface callers rely on (Python API, TOML, generated command, or result file).
- Before editing, share a short test plan: the input or action, the expected observable result, and how to check it. For a bug, reproduce the failure when practical. Proceed with the change.
- Reuse the existing representation and extension point where they fit. Introduce another type when it has a distinct owner, lifetime, or invariant. Keep standard interfaces usable by their current callers.
- Extend the closest existing test or fixture. Add a case when it catches a distinct regression. Exercise the normal entry point and report what ran, what it showed, and any claim that still needs a live workload.
