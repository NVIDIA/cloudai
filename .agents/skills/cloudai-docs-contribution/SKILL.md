---
name: cloudai-docs-contribution
description: Write, edit, or review CloudAI documentation, including README.md, doc/, release notes, and docstrings.
---

# CloudAI docs contribution

- Write for the intended reader, not the PR reviewer. User docs explain how to use CloudAI;
  implementation and packaging details belong only where they help the reader's task.
- Describe behavior in the revision being documented. Keep PR stages, roadmap promises and proof-of-concept
  narration out of product docs; put change history in release notes.
- Explain supported behavior rather than listing everything not implemented. Deliberate product scope is not
  a known issue. Include constraints when readers need them to use the feature correctly.
- Cut obvious benefits, boilerplate introductions and sentences that repeat the heading. Be concise without
  becoming cryptic: distinguish test cases, iterations and DSE steps instead of calling everything a "run".
- Extend the relevant page and link existing explanations instead of duplicating them. Keep usage instructions,
  reference details and implementation rationale distinct; no new page hierarchy for a small feature.
- Prefer a small concrete example over vague prose. Show the actual option, input or output and explain its
  meaning; don't turn examples into exhaustive schemas or invent behavior to make an explanation sound complete.
- Docstrings should explain non-obvious behavior, not restate names. For an unclear transformation, a short
  input/output example is more useful than a paragraph describing the implementation.
- Check the rendered result when changing docs: list nesting, code blocks and links can look correct in source
  but fail in the published page.
