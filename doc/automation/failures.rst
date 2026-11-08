Handle results and failures
===========================

Inspect the experiment status before using its measurements. Preserve failed runs as evidence rather than silently treating missing metrics as zero.

Load saved evidence
-------------------

.. code-block:: python

   from pathlib import Path
   import cloudai.api

   experiment = cloudai.api.get_experiment(Path("results/your-experiment-directory"))
   print(experiment.status)

Replace the path with the directory printed by execution. Use ``list_experiments(system)`` to discover saved experiments.
See :doc:`/reporting` for the record structure and :doc:`/api` for lookup errors.

Distinguish failures
--------------------

Configuration and setup errors raise exceptions. Workload failures appear in the returned experiment's status.
Missing metrics need investigation even when a process completed. Store logs and the generated configuration so
another process or person can diagnose the cause.

Cancellation
------------

Pass ``on_start`` to receive the initial experiment and cancellation function before execution begins.
The call remains synchronous; the callback is not an asynchronous execution API. Follow the contract in :doc:`/api`
when integrating with an application's cancellation handling.
