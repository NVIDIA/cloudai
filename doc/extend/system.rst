Add an execution system or runner
=================================

Add a backend when CloudAI must execute experiments through another scheduler or environment.

Follow an existing backend
--------------------------

Read ``src/cloudai/systems/standalone`` for local process execution and ``src/cloudai/systems/slurm`` for scheduled
execution. Identify the system model, runner, installer, and workload execution strategies before writing a new backend.

Connect the components
----------------------

* Define the system's configuration through the public core interfaces.
* Implement submission, polling, completion, cancellation, and result-path handling in the runner.
* Provide prerequisite installation appropriate to the environment.
* Register the system, runner, and installer with ``Registry`` in the startup registration path.
* Register workload strategies for that system and declare their compatibility explicitly.

Import shared interfaces from ``cloudai.core`` and respect the core's dependency boundaries.
An execution backend and a workload adapter solve different parts of the task; verify both together.

Verify behavior
---------------

Cover submission failure, queued and running states, completion, cancellation, and output collection.
Document path visibility, resource allocation, authentication, and supported workloads. Preserve the structured
experiment output consumed by :doc:`/automation/index`. Follow :doc:`contribute` for repository conventions.
