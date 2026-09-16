---
name: cloudai-remote-install
description: Install or update a CloudAI checkout on a remote cluster for Slurm or standalone execution. Does not run benchmarks.
---

## Installation

- One local checkout -> one remote installation with its own uv environment.
  Use remote home for Python projects and environments, not NFS/Lustre.
- Deploy the main local checkout to `~/cloudai`. Never delete it or deploy a
  linked worktree over it. Keep it maintained without overwriting remote edits.
- Deploy linked worktrees to `~/cloudai-worktrees/<checkout-id>`.
  Identify the checkout, not its branch.
- Transfer only files needed for installation and the task, including local
  changes. Exclude local environments, caches, bytecode and Git metadata.
  Git-based deployment is also fine.
- Use [scripts/deploy.py](scripts/deploy.py) for deployment.
- Do not update an installation while running or queued work still uses it.
- The script overwrites selected files but does not delete remote files; remove obsolete source files explicitly.

## Cluster configuration

- Reuse the user's existing system config across installations on the same
  cluster. Keep shared configs outside deployment directories.
- Different execution backends may need different configs, but configs on
  the same cluster should share artifact installation and results paths.
  Use NFS/Lustre for these paths.
- Look for existing config and storage hints in the user's SSH configuration
  and cluster setup. If unclear, propose concrete paths and ask the user.
- Persist `CLOUDAI_SYSTEM_CONFIG` in the appropriate remote shell startup file.
  Feature-specific configs should be explicit overrides, not replacements
  for the user's default. Pass the selected config to noninteractive commands
  too; they may not load the shell startup file.

## Cleanup

- During remote work, check for deployments unused for more than 21 days.
  Touch `.cloudai-last-used` in the deployment when using it; the helper does
  this after installation. Missing markers or unrecorded manual use are not
  proof of inactivity.
- Ask before removing each candidate, after checking running and queued jobs.
  Never remove `~/cloudai`, active or queued-job deployments, or shared configs
  and artifacts.
- If cleanup is declined, place `.cloudai-keep` in that
  deployment and exclude it from future cleanup suggestions.
