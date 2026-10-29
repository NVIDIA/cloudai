---
name: pr-management
description: Use after every push to a branch with an open PR, and when creating or updating PRs. Maintain title, description, labels, and assignee.
---

# PR management

Honor existing authorization. Text-only request: draft text, suggest metadata, no GitHub writes.
Create PRs as drafts (`gh pr create --draft`). Mark ready only when asked; preserve existing PR status.

## After every push

- After each successful authorized push, find the open PR matching the pushed repository and head branch. Apply this skill before reporting completion, including for commit-and-push requests.
- Push authorization includes maintaining that PR's title, description, labels, and assignee. No separate approval needed. No open PR: stop; do not create one unless asked.
- Read current PR and final diff. Update stale metadata; preserve accurate user edits. Check CI for the new head; never carry forward an older commit's passing status. Already current: no write needed.

## Description

- Follow repository template; match user's edited descriptions. PR prose: normal sentences, concise.
- Short title: component + change, e.g. `[MegatronRun] Metric observations`.
- Summary: usually one sentence covering user-visible result and affected interface/artifact.
- Explain final behavior. Skip supporting-change inventories: helpers, tests, configs, docs, refactors unless essential to review.
- Match current diff. Remove abandoned approaches, stale claims, reassurance about unchanged behavior.
- Additional Notes: `N/A` unless compatibility change, material limitation, or reviewer action.

## Test plan

- Green CI covering routine checks: `CI passed.` Enough. Verify current-head status before claiming success.
- Do not enumerate unit tests, test cases/counts, pytest commands, pre-commit hooks, formatting, lint, type checks, or local environment details already covered by CI. No duplicate checklists elsewhere in description.
- Pending, failed, or unavailable CI: state accurately and briefly. Include actionable failures or material coverage gaps; never imply success.
- Add distinctive evidence CI cannot provide: real installation, remote workload, performance measurement. Keep setup, outcome, and relevant measurement concise.
- More detail only when explicitly requested or specifically required by repository instructions. A Test Plan heading alone does not require commands or check inventories.
- Reporting rule only: still run required checks.

Example: `CI passed. Manual Slurm import completed in 34 seconds with a 20-minute limit.`

## Labels

- Inspect repository label definitions and current PR labels. Add few relevant labels; preserve existing labels and exact casing. Create labels only when requested.
- Label main change. Supporting tests/docs alone do not justify extra labels.
- `compatibility-change`: release-changelog marker for upgrade-relevant config, CLI, integration API, or output changes. Includes changed behavior even when old configs still parse; excludes purely private refactors.
- Explain affected users/integrations, before/after behavior, migration, and whether action is required in Additional Notes or equivalent.
- Reassess after scope changes. Flag stale existing labels; do not silently remove. Missing `compatibility-change`: include note, suggest label.

## Assignee

- Read actual PR author and assignees. No assignees: assign author's GitHub login. Any assignee present: preserve entire list.
- Recheck immediately before assigning. Author unavailable for assignment: report; no substitute.
