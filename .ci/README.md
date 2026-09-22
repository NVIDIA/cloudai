# CloudAI PR CI

Blossom Jenkins pipeline that runs on every pull request.

| File | Role |
| --- | --- |
| `proj_jjb.yaml` | Job definition, applied with `jenkins-jobs update` |
| `Jenkinsfile` | Thin entry point; reports commit status, hands off to ci-demo |
| `job_matrix.yaml` | The stages that actually run |
| `header-check.yml` | Copyright-header policy for `header_check.py` |

Resulting job: `CloudAI` / `cloudai-ci`, alongside the existing
`cloudai-release` job.

## Flow

`.github/workflows/blossom-ci.yml` authorizes the requester, runs Blossom's own
vulnerability scan, then triggers this job via the `CI_SERVER` secret
(`<jenkins-url>@cloudai-ci`). Jenkins checks out the PR merged into its base
branch and runs the matrix.

## Stages

- **Check copyrights** — `header_check.py` against `header-check.yml`, limited
  to files touched this calendar year.
- **Secret scan** — `secret_scan.py` over the working tree.
- **Coverity scan** — `--all-security`, reports only. Gating on findings needs
  a Coverity stream to diff against; without one every pre-existing defect
  would block every PR.
- **Coverage** — `pytest --cov`. Containers run as root, so the two
  permission-sensitive tests are run separately with their exit code inverted.

Blackduck and the antivirus scan stay in the release pipeline: Blossom already
Blackducks the PR before this job starts, and the antivirus stage only acts on
published release tarballs.

GitHub Actions (`.github/workflows/ci.yml`) still owns linting, type checking,
the docs build and the install smoke test.

## Changing CI

Everything this job does lives in `job_matrix.yaml`, so changing CI behaviour
is an ordinary pull request. Only three things need Jenkins-side action, and
all are one-time:

1. `jenkins-jobs update .ci/proj_jjb.yaml` to create or update the job
2. installing credentials
3. adding namespace egress rules when a stage needs a new host

Do not edit the job through the Jenkins web UI — changes are overwritten on the
next update.
