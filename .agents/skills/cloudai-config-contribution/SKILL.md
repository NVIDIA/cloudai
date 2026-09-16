---
name: cloudai-config-contribution
description: Create, adapt, review, or troubleshoot CloudAI system, test, and test-scenario TOML configurations. Use for CloudAI config-file requests, not for implementing workload Python code.
---

# CloudAI config contribution

- CloudAI is a public repo thus configs must contain no internal references
- `conf/experimental` is the place to put configs
- prefer test-in-scenario. Use `path`-based references in scenario when a single test TOML may serve different scenarios
