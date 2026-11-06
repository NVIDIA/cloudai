# Publishing documentation

## Deployment flow

| Event | Source | Destination |
| --- | --- | --- |
| Push to `main` | Exact triggering commit | Production `dev` |
| Published non-prerelease GitHub Release | Exact tagged commit | Production |
| Published prerelease | None | No deployment |
| Manual **Publish Fern Documentation** | Selected branch, tag, or SHA | Preview by default; production can be selected |

Manual production releases require `release_tag` and a `source_ref` resolving to that exact tag.
Manual development publishing accepts commits from `main`. Already published release versions
cannot be replaced with another commit. The highest numeric release version is the default;
maintenance releases do not move the default backward. All published releases remain listed.

Stable GitHub releases publish directly to production in one job. Review documentation beforehand
using manual previews. The job resolves the tagged commit and reads the latest
production version state, preserving published dev and other release versions.

## Where version state lives

The CI-managed `release/docs` branch contains only `fern/docs.yml` and `fern/fern.config.json`.
Its version entries pin source commit SHAs, which already contain the generated Python references.
It does not contain copies of pages. Do not merge this branch into development branches.

Each production run reads the latest configuration, updates one entry, publishes, and only then
commits the successful configuration to `release/docs`. Failed publishing leaves the registry
unchanged. If publication succeeds but saving the registry fails, rerun that job before starting
another production deployment; the Git push and Fern publication are not one atomic operation.

Preview runs read that registry and prepend a temporary candidate entry, making it the preview default. Existing production versions,
including the published `dev` SHA, remain unchanged. Manual previews reuse a stable URL for the selected source ref. The run summary records the source SHA and URL; workflow artifacts retain the candidate
configuration and publish log. GitHub's artifact retention policy applies to this preview history.

GitHub Actions serializes production jobs with the `fern-production` concurrency group and
`queue: max`. Previews use separate groups per branch/ref. The stale-dev check prevents an older queued commit from rolling production `dev` backward.

Fern checks out the pinned source refs to assemble the site. Publishing never runs historical
Python generators or creates generated-documentation commits on source branches.

## One-time GitHub setup

- Set the repository secret `FERN_TOKEN` for the `nvidia` organization. It serves both preview
  and production publishing.
- `fern/docs.yml` uses `nvidia-cloudai.docs.buildwithfern.com/cloudai`, following the NVIDIA sample
  naming convention, with public domain `docs.nvidia.com/cloudai`. Coordinate activation of the
  public route with the Fern administrators.
- Create `docs-preview` and `docs-hosted` environments without required reviewers.
- Exclude exactly `release/docs` from the `release/*` ruleset. CI creates and updates it with
  the standard `GITHUB_TOKEN` and `contents: write`; no deploy key or extra token is needed.
  Repository writers can also push directly to this branch.
- Merge publishing tools into `main` before using them. Every deployment reads tools and shared
  site settings from `main`; source documentation comes from the selected revision. Release branches
  need the trigger workflows, Fern configuration, and committed generated references.

The first production run creates the configuration branch, seeded with the historical version refs
in `fern/docs.yml`. The 1.6.1, 1.7.1, and 1.8.1 backfills contain static MDX, including Python references;
1.8.1 uses the release branch revision tagged `v1.8.1-2`. These snapshots need no generation during publishing.

Fern local preview renders only working-tree versions. To preview a historical version locally,
check out its `fern-backfill/<version>` branch and run the Fern CLI there. The combined ref-backed
site is assembled when publishing.
