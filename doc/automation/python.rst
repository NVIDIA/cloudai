Run experiments with Python
===========================

Use CloudAI as an execution step in your own script or research agent. Supply a system and scenario, then inspect the returned experiment.

Start with a known scenario
---------------------------

Prepare the Standalone Fio setup in :doc:`/Tutorial`, then run this from the repository root with ``uv run``:

.. code-block:: python

   from pathlib import Path
   import cloudai.api

   system = Path("standalone.toml")
   scenario = Path("conf/experimental/quickstart/fio.toml")
   cloudai.api.validate_scenario(scenario, system)
   experiment = cloudai.api.run_experiment(scenario, system)
   print(experiment.status, experiment.path)

``run_experiment`` is synchronous: it returns after execution finishes. The returned model contains the data saved
in ``experiment.json``. An external agent can use that evidence to choose the next scenario.

Integrate into a larger investigation
-------------------------------------

Keep the task's reasoning and experiment selection in your application. Use CloudAI to validate and execute the chosen
workloads. Record the scenario and result directory for each decision. Use :doc:`failures` to distinguish execution
failures from configuration errors, and :doc:`/api` for the exact function contracts.
