Python API
==========

Use ``cloudai.api`` to validate scenarios, run experiments, and read their results
from Python. The CLI and API share the same execution logic.

Run an experiment
-----------------

Pass configuration files as ``pathlib.Path`` objects:

.. code-block:: python

   from pathlib import Path

   import cloudai.api

   scenario = Path("conf/common/test_scenario/sleep.toml")
   system = Path("conf/common/system/example_slurm_cluster.toml")
   tests_dir = Path("conf/common/test")

   valid, errors = cloudai.api.validate_scenario(scenario, system, tests_dir=tests_dir)
   if not valid:
       raise ValueError(errors)

   experiment = cloudai.api.run_experiment(scenario, system, tests_dir=tests_dir)
   print(experiment.status, experiment.path)

A ``str`` argument for ``scenario`` or ``system`` means TOML content. Use ``Path``
for a filename. Test ``path`` references resolve relative to a scenario file;
when passing scenario text, use absolute test paths or inline test definitions.
Other relative paths keep their usual meaning relative to the working directory.

``run_experiment`` installs workload prerequisites as needed and returns an
``Experiment`` model containing the saved results. Each invocation gets its own
result directory. Use ``experiment.model_dump()`` or ``experiment.model_dump_json()``
to serialize the result.

Optional keyword arguments:

* ``tests_dir`` and ``hook_dir`` select test definitions and hooks.
* ``output_dir`` overrides the system's results directory.
* ``dry_run=True`` generates workload commands without submitting jobs.
* ``single_sbatch=True`` runs a Slurm scenario in one allocation.
* ``enable_cache_without_check=True`` uses installed workload components without
  checking them first, as with the CLI option.

Background execution
--------------------

Set ``wait=False`` to launch a separate worker process and return a pending
experiment. The worker continues after the calling process exits. Read the latest
saved snapshot using the returned path:

.. code-block:: python

   experiment = cloudai.api.run_experiment(scenario, system, wait=False, tests_dir=tests_dir)
   # Later, including from another process:
   experiment = cloudai.api.get_experiment(experiment.path)
   print(experiment.status)

The result directory contains ``experiment.json``, the worker's ``controller.log``,
and its ``request.json``. Configuration supplied as text is saved there as TOML.
Keep referenced config files available until the worker has read them. The worker
loads the installed CloudAI package and plugins in a fresh Python process;
registrations made only in the caller's memory are not transferred.

Find and read results
---------------------

``list_experiments(system)`` returns ``(experiment_id, result_directory)`` pairs
from the system's ``output_path``, sorted by directory name. Directories without
``experiment.json`` are skipped. A missing results directory returns an empty list.
When using ``output_dir`` for a run, use that directory as ``output_path`` in the
system configuration passed to ``list_experiments``.

``get_experiment(exp)`` accepts a result directory or the ``experiment.json`` file
itself, as either ``str`` or ``Path``. It reads the saved snapshot without querying
the scheduler. See :doc:`reporting` for the result fields.

Errors
------

``validate_scenario`` returns ``(True, {})`` for valid inputs, or ``(False, errors)``
with ``system`` or ``scenario`` keys and error messages. It checks configuration and
execution constraints without installing workloads, creating a results directory,
or querying cluster availability.

``run_experiment`` raises exceptions for invalid configuration and setup errors.
Workload failures appear in the experiment and run statuses. For background runs,
errors after launch are recorded as a failed experiment, with details in
``controller.log``.

Missing result files raise ``FileNotFoundError``. Invalid experiment JSON raises
``pydantic.ValidationError``. Listing also raises for unreadable or invalid result
files rather than silently omitting them.

The API does not install signal handlers or invoke the CLI's logging configuration.
The legacy public functions in ``cloudai.cli.handlers`` remain available with
deprecation warnings; use ``cloudai.api`` for new integrations.
