# Documentation

Documentation is authored in `doc/*.mdx` and `doc/workloads/*.mdx` and built with Fern.
Keep the root README separate. Navigation lives in `fern/docs.yml`.
Paths and commands below are relative to the repository root.

With Python 3.14, Node.js 24 and uv installed, run `make -C doc check` to generate the Python reference
and validate the documentation, or `make -C doc preview` to start a local preview.
After changing Python docstrings or model fields, rerun `make -C doc reference`.
The preview uses Fern's legacy renderer because the default Astro preview currently returns
404s for compiled MDX modules, causing page content to disappear after loading.

Generated reference snippets live in `fern/generated/` and are not checked in. Add classes
and member exclusions to `fern/reference.yml`; include their snippets in authored pages with
`<Markdown src="/generated/fully.qualified.Class.mdx" />`. The build reads Python objects
and renders `doc/templates/reference.mdx.jinja`; it does not transform authored MDX.

CI validates the docs and produces a Fern source artifact. Publishing requires the provisioned
Fern instance URL (`FERN_DOCS_URL`) and the enterprise deployment environment. The default URL
is for local validation only; this repository does not publish automatically.
