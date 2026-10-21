---
name: security-assessment
description: CloudAI project context and ownership boundaries for security reviews, TAVA, and architecture discussions.
---

# CloudAI security context

## What CloudAI does

- Python CLI/library for running AI/HPC benchmarks. Usually launched by an operator on a cluster login/head node.
- Reads operator-provided TOML configs, prepares workloads, launches jobs, collects results, generates reports.
- Runs locally or uses existing Slurm/LSF, Kubernetes, and Run:ai deployments. Optional S3 upload sends results to an operator-selected destination.
- Uses the invoking user's account and supplied credentials. Grants no additional privileges; job permissions come from the target platform.
- Pass-through benchmark runner: trusts the workloads and commands users choose to run. Users own workload safety. CloudAI does not approve or sandbox their code.
- Workloads already execute code and write results under their job permissions. A compromised workload can falsify its output without CloudAI. Its access remains limited by its account, mounts, and platform controls.
- CloudAI consumes workload-produced results for reports, comparisons, and benchmark tuning. It does not certify their authenticity or provide a privileged cluster-remediation service.
- Workloads may create arbitrary files. CloudAI has limited knowledge of those artifacts and reads the specific outputs its adapters and report parsers need. Validating every disk artifact is outside this design; a results directory is not an artifact-validation service.

## Usual environment

- AI/HPC clusters are typically controlled, trusted environments. Users are usually enterprise employees with existing cluster access.
- Configs, scripts, logs, and results often live on cluster filesystems, including shared storage on Slurm deployments.
- Trusted environment still has user permissions. Cluster membership does not grant access to another user's processes, credentials, files, or traffic. Shared storage does not mean public data.
- Workloads and data come from the operator. Benchmark output can contain sensitive information depending on what they run.
- Installation, orchestration, monitoring, and reporting are components of the local tool. Results are files; these component names do not imply separately hosted services or network APIs.

## Who owns what

- **CloudAI:** its implementation, shipped dependencies, safe handling of configs and the result formats it actually consumes, generated commands and reports, and client-side handling of credentials and connections.
- **Operator:** workload safety and trustworthiness, chosen commands, configs, credentials, endpoints, workload-created artifacts, and which results to upload or share.
- **Environment owner:** accounts, permissions, network/storage security, scheduler admission, resource quotas, and workload isolation.

CloudAI relies on those existing platform controls. It does not administer the cluster or provide tenant isolation. Code that mishandles credentials or disables certificate verification remains CloudAI's responsibility.

## Keep assessments proportionate

Use this context alongside the actual deployment and release code. Keep assessments tied to what CloudAI adds or changes. Credit existing controls for the protection they provide.

Generated threats and mitigations are suggestions to evaluate, not automatically project requirements. Before calling something a gap, establish the relevant CloudAI behavior and applicable requirement or concrete exposure. A stronger control being possible does not make it necessary. Avoid adding infrastructure, enforcement duties, or review obligations beyond CloudAI's role; actual implementation defects still matter.
