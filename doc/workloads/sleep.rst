Sleep
=====

This workload (`test_template_name` is ``Sleep``) executes a simple sleep command for testing and timing purposes. Useful for testing schedulers and system behavior.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_sleep_test"
   description = "Example Sleep test"
   test_template_name = "Sleep"

   [cmd_args]
   seconds = 30

Test Scenario example:

.. code-block:: toml

   name = "sleep-test"

   [[Tests]]
   id = "sleep.1"
   num_nodes = 1
   time_limit = "00:02:00"

   test_name = "my_sleep_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "sleep-test"

   [[Tests]]
   id = "sleep.1"
   num_nodes = 1
   time_limit = "00:02:00"

   name = "my_sleep_test"
   description = "Example Sleep test"
   test_template_name = "Sleep"

     [Tests.cmd_args]
     seconds = 30

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.sleep.sleep.SleepCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.sleep.sleep.SleepTestDefinition
   :members:
   :show-inheritance:
