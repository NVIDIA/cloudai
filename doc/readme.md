# Python reference generation

Pydantic field tables are generated from the models listed in `fern/reference.yml` and included
in the MDX pages. Edit the models or `doc/generate_reference.py`, not `fern/generated/`.

Generate snippets with Python 3.14:

```sh
uv run --locked --extra docs python doc/generate_reference.py
```

Pre-commit runs the generator too. Stage changed and newly generated snippets, then retry.
