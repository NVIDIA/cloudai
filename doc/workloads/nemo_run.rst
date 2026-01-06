Nemo Run
========

This workload (`test_template_name` is ``NemoRun``) executes NeMo training and fine-tuning tasks using the NeMo Run framework.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_nemo_test"
   description = "Example NeMo Run test"
   test_template_name = "NemoRun"

   [cmd_args]
   recipe = "llama3_8b"
   task = "pretrain"

Test Scenario example:

.. code-block:: toml

   name = "nemo-run-test"

   [[Tests]]
   id = "nemo.1"
   num_nodes = 4
   time_limit = "02:00:00"

   test_name = "my_nemo_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "nemo-run-test"

   [[Tests]]
   id = "nemo.1"
   num_nodes = 4
   time_limit = "02:00:00"

   name = "my_nemo_test"
   description = "Example NeMo Run test"
   test_template_name = "NemoRun"

     [Tests.cmd_args]
     recipe = "llama3_8b"
     task = "pretrain"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nemo_run.nemo_run.NeMoRunCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.nemo_run.nemo_run.NeMoRunTestDefinition
   :members:
   :show-inheritance:
