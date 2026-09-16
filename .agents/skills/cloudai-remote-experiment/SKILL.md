---
name: cloudai-remote-experiment
description: Run, monitor, stop, and retrieve a CloudAI experiment (test scenario) on a remote Slurm or standalone cluster.
---

## Run

- Use [cloudai-remote-install](../cloudai-remote-install/SKILL.md) first. Identify
  this checkout's remote installation and system config; do not use another checkout's deployment.
  Use its wrapper for all remote commands and transfers, not direct SSH.
- Run the CloudAI controller as a detached daemon that survives disconnects,
  with stdin closed and output redirected. Keep separate per-run controller/debug
  logs in remote home, not Lustre. Record the actual controller PID, start time,
  deployment, selected configs, exact results directory and eventual exit status.
- Establish expected progress and a workload-specific stall timeout before launching.
  Include legitimate silent phases such as initialization; ask if expectations are unclear.

## Monitor and stop

- Stay responsible for monitoring until the run finishes or is explicitly handed back.
  Check the recorded process with `ps` and read bounded log increments no more than
  once per minute; back off when unchanged. Sample only a few known artifacts for progress.
  No recursive scans, whole-log rereads or continuous result syncing on shared storage.
- Never use `squeue --me` or equivalent user-wide queue queries. Let CloudAI poll
  Slurm; do not add scheduler polling loops. If diagnosis requires scheduler state,
  make a targeted query for this run's recorded job IDs only.
- Track submitted job IDs from this run's own log. Completion metadata can corroborate
  ownership but may not exist while jobs are running. Never infer ownership from
  username, job name or checkout alone: a checkout may have several runs.
- Silence alone is not a stall, and queued jobs are not wasting an allocation.
  Stop confirmed stuck work after its stall timeout, using repeated evidence of
  missing expected progress in an active allocation. Ask if the evidence is ambiguous.
- Before stopping, recheck process identity and job ownership. Gracefully terminate
  only this run's controller and stop further submissions; then `scancel` its remaining
  exact allocation IDs, including queued jobs. Verify termination and allocation release.
  For standalone runs, stop only this run's process tree. Never use user-wide cancellation
  or broad process-name matching. Preserve logs explaining why the run was stopped.

## Results

- Confirm completion from logs and job outcomes, not just a vanished PID or zero exit code.
  After success, failure or cancellation, copy the exact run directory into
  `results/<cluster-nickname>/<run-directory>/` locally, without overwriting another run.
- Derive the cluster nickname from the selected config and SSH target, not blindly
  from the config name: `<cluster>-for-testing` still belongs under `<cluster>`.
  Ask if ambiguous.
- Stream an archive through the wrapper's `run` action for retrieval (`copy` uploads only).
  Copy this run's output, not the shared results tree; preserve remote files.
  Report the outcome and local results path, including any incomplete retrieval.
