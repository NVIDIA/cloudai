Compose a scenario
==================

Start with one case, then add independent or coordinated workloads to the same scenario.

One case
--------

Copy a scenario from ``conf/experimental/quickstart`` and run it with its compatible system.
Follow :doc:`configuration` for inline definitions and reusable test files.

Several cases
-------------

Add a ``[[Tests]]`` table with a unique ``id`` for each case. Independent cases may execute concurrently;
account for their combined resource requirements. Use dependencies when execution order matters.

For a case that must follow another, add this table within its ``[[Tests]]`` entry:

.. code-block:: toml

   [[Tests.dependencies]]
   type = "start_post_comp"
   id = "first-case"

Replace ``first-case`` with the preceding case's ID. The dependency syntax and other dependency types are described in
:doc:`/reference/scenarios`. Use :doc:`hooks` for repeated preparation and cleanup.

Configuration searches
----------------------

Keep a baseline scenario and a DSE scenario separate. The ordinary DSE execution path rejects dependencies and
mixed DSE/non-DSE cases. See :doc:`/optimization/search` for a minimal search.
Validate and inspect generated commands with :doc:`preview` before requesting resources.
