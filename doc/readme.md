# Documentation

Edit pages in `doc/**/*.mdx`; use lowercase, hyphenated filenames.
`fern/dev.yml` defines navigation and page paths for this revision. Releases use the file from their pinned commit.
`fern/docs.yml` holds site settings. The root README is separate from the Fern site.

## Local preview

Install Python 3.14, uv, and Node.js 24, then run:

```bash
make -C doc preview
```

This uses Fern's legacy renderer; check the changed pages and links in the browser.

## Page structure

Start with the task and a small working example; reveal adaptations and reference details as readers need them.
Use Fern `Tabs` for alternative scenarios, `Accordion` for optional configurations, `Steps` for sequential tasks,
and `CardGroup` for navigation choices. Keep the landing page minimal and the root README self-contained.
Add each page to `fern/dev.yml`, including category overview pages. Keep starter snippets aligned with
`conf/experimental/quickstart/` and the workload/system matrix aligned with the registry and README.

## Python references

Register Pydantic models in `fern/reference.yml`, then run `make -C doc reference`.
Include the generated field table in the relevant page:

```mdx
<Markdown src="/generated/cloudai.workloads.sleep.sleep.sleepcmdargs.mdx" />
```

Edit the model or generator, not files in `fern/generated/`. Pre-commit regenerates the tables;
stage any changed or newly generated files and retry the commit. CI checks the same hook.
Methods and non-Pydantic references stay in authored pages.

Use base-path-free links such as `/guides/slurm`; Fern adds the configured site prefix. Unversioned URLs
lead to the default stable version.

For deployment configuration and version management, see [Publishing](publishing.md).
