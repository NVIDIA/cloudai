Develop and contribute
======================

Make a focused change with a public example and validation appropriate to its behavior.

Prepare
-------

Follow the repository's `contribution guidelines <https://github.com/NVIDIA/cloudai/blob/main/CONTRIBUTING.md>`_.
Use SPDX headers for new Python files, mirrored tests, absolute core imports, and existing lazy-loading patterns.
Keep configurations backward compatible and examples free of internal resources and credentials.

Check a change
--------------

.. code-block:: bash

   uv run --locked --extra dev pytest tests/path_to_changed_behavior.py
   uv run --locked --extra dev pre-commit run --files path/to/changed_file.py

Replace the paths with the tests and files affected by your change. Build documentation with
``uv run --extra docs sphinx-build -b html doc doc/_build/html`` and inspect the rendered pages.

Submit
------

Sign off commits and follow the pull-request template. Describe the user-visible behavior and validation performed.
Update the relevant task guide or reference when public behavior changes. Keep runtime examples and their documentation
linked so later changes can be validated together.
