Python API
==========

``cloudai.api`` lets a Python program run a scenario and find its saved results.
Pass configuration file paths as ``pathlib.Path`` objects:

.. code-block:: python

   from pathlib import Path

   import cloudai.api

   system = Path("conf/common/system/example_slurm_cluster.toml")
   scenario = Path("conf/common/test_scenario/sleep.toml")

   experiment = cloudai.api.run_experiment(
       scenario, system, tests_dir=Path("conf/common/test")
   )
   print(experiment.status, experiment.path)

``run_experiment`` waits for the scenario to finish and returns an ``Experiment``
model with the same data saved in ``experiment.json``. Workload failures appear in
its status. Configuration and setup errors raise exceptions. Use
``mode="dry-run"`` to generate commands without running workloads. Use
``single_sbatch=True`` for a single Slurm allocation. Pass ``hook_dir`` to use a
custom hook directory. Pass an ``on_start`` callback to receive the initial
``Experiment`` and a cancellation function after ``experiment.json`` is created
and before execution begins. The function remains synchronous after calling the
callback.

``list_experiments(system)`` returns ``(experiment_id, result_directory)`` pairs
from the system's configured output directory. Directories without
``experiment.json`` are skipped. A missing output directory gives an empty list;
experiment file contents are not parsed.

``get_experiment(experiment_id, system)`` loads an ``Experiment`` from that
system's output directory. Pass a ``Path`` to a result directory to load it
directly. Invalid IDs and missing, unreadable, or malformed experiment files
raise ``ValueError``.

Models for workload extensions
------------------------------

Import ``TestScenario``, ``TestDefinition`` and ``TestRun`` from ``cloudai.core``.
Workload definitions continue to inherit ``TestDefinition`` and generation
strategies keep receiving ``system`` and ``test_run``.

``TestScenario.tests`` contains the resolved workload definitions. Each definition
owns its workload arguments and configured scheduling, repetitions and dependency
IDs. ``name`` identifies the reusable test; ``id`` identifies its occurrence in a
scenario. Standalone definitions can omit ``id``.

Before execution, prepare a separate ``TestDefinition`` with the selected sweep
values and effective scheduling settings. Keep the scenario's original definition
unchanged. Treat prepared definitions as immutable configuration; create a new
definition when settings change.

``TestRun(test=prepared_definition)`` holds that definition directly. It stores
allocated nodes, iteration, step, output path and resolved dependency and hook
bindings. Static settings, including selected node count, effective time limit,
weight and SOL, belong to ``TestRun.test``. Existing read access through
``test_run.nnodes``, ``iterations``, ``pin_nodes`` and ``extra_srun_args`` forwards
to that definition.

``TestDefinition.sol`` and ``TestRun.sol`` contain either a number, a ``SOLMetricModel`` or ``None``.
Import ``SOLMetricModel`` from ``cloudai.core``; its ``root`` maps metric names to target lists. Structured SOL
serializes directly as the existing metric-keyed ``sol`` tables. ``System.sol`` and ``TestScenario.sol`` use that
same model. ``TestRun.metric_sol`` is removed.

The existing runners use ``TestScenario.test_runs`` for bound executions.
Replacing that list does not replace ``TestScenario.tests``. The execution list
is excluded from scenario serialization.

``test-run.toml`` and best-config exports include the full test definition,
including scheduling configuration. Existing saved runs that omit those fields
remain readable.

Migration notes for direct model users:

* ``TestRunModel`` and ``TestScenarioModel`` are removed. Use ``TestDefinition``
  and ``TestScenario``. Pass unresolved ``test_name`` or ``path`` references
  through the existing parsers, which resolve them before model validation.
  ``TestRunDependencyModel`` now lives in ``cloudai.models.workload``.
* ``get_reporters`` in ``cloudai.test_scenario_parser`` now takes only the
  workload definition; its unused scenario-item argument is removed.
* ``TestRun`` and ``TestScenario`` are no longer dataclasses. Dataclass field
  introspection and generated structural equality are not retained.
* ``TestRun`` takes a prepared definition and runtime state. Configuration
  arguments such as ``name``, ``num_nodes`` and ``iterations`` move to the
  definition (``name`` becomes the case ``id``). Static forwarding properties are
  read-only. ``TestRun.definition`` is removed; ``TestRun.test`` is its sole
  definition reference. The constructor no longer copies the definition.
* ``extra_srun_args`` accepts a string or a list of string fragments in test and
  scenario TOMLs. Definitions store a list; ``TestRun.extra_srun_args`` provides
  the joined string for existing strategies. Existing quoting is preserved.
  Scenario-case arguments now replace test-file arguments instead of being
  appended to them; an empty string or list clears inherited test arguments.
  SlurmContainer test-file arguments now also apply to metadata commands.
  See :ref:`extra-slurm-arguments` for configuration rules and examples.
