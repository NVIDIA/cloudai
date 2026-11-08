Add a workload
==============

Add a workload when an existing integration cannot execute the task you need. Start with a nearby implementation and preserve CloudAI's common execution behavior.

Choose the smallest extension
-----------------------------

For an existing script or container, first consider :doc:`/workloads/bash_cmd` or :doc:`/workloads/slurm_container`.
For a reusable integration, define its configuration and the execution strategy for each supported system.

Implement and register
----------------------

1. Define command arguments and a test definition using the public interfaces in ``cloudai.core``.
2. Implement command generation for the intended system. Reuse base strategies for allocations, mounts, and hooks.
3. Declare required installables and expose metrics from workload output where applicable.
4. Register the definition and strategies in ``src/cloudai/registration.py`` using ``Registry``.
5. Add a public scenario, focused tests, and a workload guide. Update the compatibility catalog.

Use ``src/cloudai/workloads/fio`` for an integration with Standalone and Slurm execution, or
``src/cloudai/workloads/nccl_test`` for collective communication. Inspect the current registry signatures before
adding registrations; a workload name alone does not register support for every scheduler.

Design for adaptation
---------------------

Pass workload options through where practical. Keep scheduler orchestration in the execution strategy and metric
interpretation in the reporter. Use mounts, installables, and hooks for environment preparation rather than hard-coding
cluster details. Do not change the meaning of existing user configurations.

Verify
------

Mirror new Python modules under ``tests``. Validate a minimal scenario, inspect generated commands, and cover distinct
execution shapes and failure behavior. Test only the schedulers declared as supported. Follow :doc:`contribute`
and :doc:`reporter` for contribution and metric integration details.
