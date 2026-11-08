Architecture and extension points
=================================

CloudAI separates configuration, execution, and result interpretation. Extend the layer responsible for the behavior you need.

.. list-table::
   :header-rows: 1

   * - Extension
     - Start here
   * - Workload configuration and execution
     - :doc:`extend/workload`
   * - Metrics and reports
     - :doc:`extend/reporter`
   * - Configuration selection
     - :doc:`extend/agent`
   * - Scheduler or execution environment
     - :doc:`extend/system`

Read :doc:`extend/contribute` before changing shared interfaces. Keep public core APIs accessible through ``cloudai.core``.

Core Modules
------------

We use `import-linter <https://github.com/seddonym/import-linter>`_ to ensure no core modules import higher level modules.

``Registry`` object is a singleton that holds implementation mappings. Users can register their own implementations to the registry or replace the default implementations.

Cache
-----

Some prerequisites can be installed. For example:

Docker images, git repos with executable scripts, etc. All such "installables" are kept under the system's ``install_path``.

Installables are shared among all tests. Therefore, if any number of tests use the same installable, it is installed only once for a particular system TOML.

``PythonExecutable`` installs a Python project into its own virtual environment. When the selected
``pyproject.toml`` has an adjacent ``uv.lock``, it uses bundled uv to install the locked base dependencies
and the project, without extras or dependency groups. An outdated lockfile fails installation instead
of being updated. Projects without a lockfile continue to use pip. When ``dependencies_from_pyproject=False``
selects an existing ``requirements.txt``, that selection takes precedence over the lockfile.
