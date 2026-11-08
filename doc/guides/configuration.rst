Understand experiment configuration
===================================

Define where to run and what to execute in two TOML files. Start with an inline workload; separate reusable definitions later.

Two files
---------

.. list-table::
   :header-rows: 1

   * - File
     - Responsibility
   * - System
     - Scheduler, resources, paths, and shared settings.
   * - Scenario
     - Test cases, workload settings, and execution requirements.

The :doc:`/Tutorial` uses ``standalone.toml`` and an inline Fio scenario:

.. literalinclude:: ../../conf/experimental/quickstart/fio.toml
   :language: toml
   :start-at: name =

``test_template_name`` selects the integration. ``cmd_args`` configures its workload.
``id`` identifies the case within the scenario; it is distinct from the scenario's name.

Reuse a definition
------------------

When several scenarios share a workload definition, move its name, template, and arguments into a test TOML.
Use ``test_name`` in each case and pass the definition directory with ``--tests-dir``.
Keep allocation settings such as ``num_nodes`` and ``time_limit`` in the scenario.

Add complexity deliberately
---------------------------

Use :doc:`scenarios` for multiple cases and dependencies, :doc:`hooks` for preparation and cleanup,
and :doc:`/optimization/search` for configuration search. List-valued settings can become search dimensions;
see :doc:`/reference/agents` before using lists as ordinary workload arguments.
Exact system fields are in :doc:`/systems`; case fields are in :doc:`/reference/scenarios`.
