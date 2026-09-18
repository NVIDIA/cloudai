---
name: cloudai-remote-install
description: Install or update a CloudAI checkout on a remote cluster for Slurm or standalone execution. Does not run benchmarks.
---

## Installation

- One local checkout -> one remote installation with its own uv environment.
  Use remote home for Python projects and environments, not the shared artifact directory.
- Deploy the main local checkout to `~/cloudai`. Never delete it or deploy a
  linked worktree over it. Keep it maintained without overwriting remote edits.
- Deploy linked worktrees to `~/cloudai-worktrees/<checkout-id>`.
  Use the first 16 hex characters of SHA-256 of local hostname + NUL + resolved
  checkout path, not the branch name. Reuse that directory on subsequent deployments.
- Transfer only files needed for installation and the task, including local
  changes. Exclude local environments, caches, bytecode, Git metadata and local results/install artifacts.
  Git-based deployment is also fine.
- Use [scripts/deploy.py](scripts/deploy.py) for all remote commands and transfers,
  not direct SSH. It only provides transport; perform the checks and setup described here.
  Invoke with Python as `deploy.py HOST [--dry-run] ACTION ...`:
  `run 'COMMAND'`, `copy '~/cloudai-worktrees/<checkout-id>/' SOURCE...`, or `fetch 'SOURCE' DEST`.
  Downloads have no upload exclusions.
- Reuse remote uv, or install it if missing. Use it to install CloudAI and manage
  the deployment's own environment; do not reuse another checkout's virtualenv.
- Do not update an installation while running or queued work still uses it.
- Preserve remote edits and personal files, including custom TOMLs. Do not prune remote-only files.
- Verify CLI startup and the selected system config after installation, without running benchmarks.

## Cluster configuration

- Reuse the user's existing system config across installations on the same
  cluster. Keep shared configs outside deployment directories.
- If none exists, inspect the cluster and adapt the closest repository example:
  cluster name, scheduler, partitions, installation/results paths and any required
  account or other cluster options. Use the current models for valid fields.
- Different execution backends may need different configs, but configs on
  the same cluster should share artifact installation and results paths.
  Use NFS/Lustre for these paths.
- Look for existing config and storage hints in the user's SSH configuration
  and cluster setup. If information is not readily available, propose what you
  can establish and ask the user rather than guessing or searching broadly.
- Persist `CLOUDAI_SYSTEM_CONFIG` in the appropriate remote shell startup file.
  Feature-specific configs should be explicit overrides, not replacements
  for the user's default. Pass the selected config to noninteractive commands
  too; they may not load the shell startup file.

## Cleanup

- During remote work, check for deployments unused for more than 21 days.
  Touch `.cloudai-last-used` after installation and whenever using the deployment.
  Missing markers or unrecorded manual use are not
  proof of inactivity.
- Ask before removing each candidate, after checking running and queued jobs.
  Never remove `~/cloudai`, active or queued-job deployments, or shared configs
  and artifacts.
- If cleanup is declined, place `.cloudai-keep` in that
  deployment and exclude it from future cleanup suggestions.
