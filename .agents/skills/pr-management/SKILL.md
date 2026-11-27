---
name: pr-management
description: Maintain PR metadata when creating or updating a PR and after each push to a branch with an open PR.
---

# Maintain a PR

- Create a draft when asked to open a PR. Mark it ready only on request.
- After each authorized push, find the open PR for that repository and head branch. If none exists, finish unless the user asked to create one.
- Compare the PR with the final diff and check CI on the new head. Update only stale metadata, preserving accurate user edits and the PR’s draft or ready state.

## Title and body

Use the repository template and contributor guide. Give the title a component and a change (for example, `[MegatronRun] Metric observations`). Summarize what changed and why. Mention implementation details only when they help the reviewer.

In the test plan, list `CI` for routine automated checks. Add a bullet for each distinct manual result, material failure, or coverage gap. Keep commands and setup details when needed to reproduce a result. For example:

- CI
- Manual Slurm import completed in 34 seconds with a 20-minute limit.

Use Additional Notes for upgrade guidance, material limitations, or reviewer action. Otherwise use `N/A`.

## Labels and assignee

- Apply existing repository labels to the main change using their definitions. Flag stale labels after a scope change. Preserve existing labels unless asked to remove them, and create labels only on request.
- Preserve assigned users. If the PR is still unassigned when updating it, assign its author by GitHub login when available.
