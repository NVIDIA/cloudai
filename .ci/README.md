# CloudAI Jenkins pipelines

Blossom Jenkins pipelines for CloudAI. Each pipeline gets its own subfolder of
the `CloudAI` Jenkins folder and its own job prefix:

| Pipeline | Jenkins jobs | Trigger |
| --- | --- | --- |
| PR | `CloudAI/PR/cloudai-pr*` | `/build` comment on a pull request |

A new pipeline `<name>` follows the same scheme: `CloudAI/<Name>/` and the
`cloudai-<name>` prefix.

| File | Role |
| --- | --- |
| `proj_jjb.yaml` | Job definitions, applied with `jenkins-jobs update` |
| `Jenkinsfile` | Runs one ci-demo matrix; reports commit status when started for a PR |
| `Jenkinsfile.launcher` | PR pipeline launcher; runs the leaf jobs in parallel |
| `pipeline/<leaf>_matrix.yaml` | The stage one leaf job runs |
| `header-check.yml` | Copyright-header policy for `header_check.py` |

## PR pipeline

Resulting jobs, in `CloudAI/PR`: the launcher `cloudai-pr` and one leaf job
`cloudai-pr-<leaf>` per matrix in `pipeline/`.

### Flow

`.github/workflows/blossom-ci.yml` authorizes the requester, runs Blossom's own
vulnerability scan, then triggers the launcher via the `CI_SERVER` secret
(`<jenkins-url>@cloudai-pr`). The launcher resolves the PR merged into its base
branch to one commit and starts every leaf job on it in parallel. Each leaf
reports its own `cloudai-pr-<leaf>` commit status on the PR, which is where a
failure shows. The launcher closes `blossom-ci`, the status the workflow leaves
pending, and fails it only if dispatching breaks. A failed leaf can be re-run
on its own in Jenkins (same parameters); its status then updates the PR.

### Stages

One leaf job per stage, each with its own matrix in `pipeline/`:

| Leaf job | Stage |
| --- | --- |
| `cloudai-pr-copyright` | **Check copyrights** — `header_check.py` against `header-check.yml`, limited to files touched this calendar year. |
| `cloudai-pr-secrets` | **Secret scan** — `secret_scan.py` over the working tree. |
| `cloudai-pr-coverity` | **Coverity scan** — `--all-security`; fails on any defect, since the codebase reports zero. |
| `cloudai-pr-coverage` | **Coverage** — `pytest --cov`. Containers run as root, so the two permission-sensitive tests are run separately with their exit code inverted. |

Blackduck and the antivirus scan stay in the release pipeline: Blossom already
Blackducks the PR before this pipeline starts, and the antivirus stage only
acts on published release tarballs.

GitHub Actions (`.github/workflows/ci.yml`) still owns linting, type checking,
the docs build and the install smoke test.

## Changing CI

Everything the leaf jobs do lives in their matrices, so changing what an
existing leaf runs is an ordinary pull request. Only three things need
Jenkins-side action, and all are one-time:

1. `jenkins-jobs update .ci/proj_jjb.yaml` to create or update the jobs. A new
   PR leaf needs this: add `pipeline/<leaf>_matrix.yaml`, list the leaf under
   `jobs` in `proj_jjb.yaml` and in `Jenkinsfile.launcher`, then update.
2. installing credentials
3. adding namespace egress rules when a stage needs a new host

Do not edit the jobs through the Jenkins web UI — changes are overwritten on
the next update.
