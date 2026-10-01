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
its status. Configuration and setup errors raise exceptions. Pass ``hook_dir`` to
use a custom hook directory.

``list_experiments(system)`` returns ``(experiment_id, result_directory)`` pairs
from the system's configured output directory. Directories without
``experiment.json`` are skipped. A missing output directory gives an empty list;
experiment file contents are not parsed.

``get_experiment(experiment_id, system)`` loads an ``Experiment`` from that
system's output directory. Pass a ``Path`` to a result directory or its
``experiment.json`` file to load it directly. A missing file raises
``FileNotFoundError``.
