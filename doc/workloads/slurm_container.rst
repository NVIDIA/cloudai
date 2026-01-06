Slurm Container
===============

This workload (`test_template_name` is ``SlurmContainer``) executes containerized workloads using Slurm with custom container configurations.

Usage Examples
--------------

Test TOML example:

.. code-block:: toml

   name = "my_container_test"
   description = "Example Slurm container test"
   test_template_name = "SlurmContainer"

   [cmd_args]
   image_path = "/path/to/container.sqsh"
   cmd = "python train.py"

Test Scenario example:

.. code-block:: toml

   name = "slurm-container-test"

   [[Tests]]
   id = "container.1"
   num_nodes = 2
   time_limit = "01:00:00"

   test_name = "my_container_test"

Test-in-Scenario example:

.. code-block:: toml

   name = "slurm-container-test"

   [[Tests]]
   id = "container.1"
   num_nodes = 2
   time_limit = "01:00:00"

   name = "my_container_test"
   description = "Example Slurm container test"
   test_template_name = "SlurmContainer"

     [Tests.cmd_args]
     image_path = "/path/to/container.sqsh"
     cmd = "python train.py"

API Documentation
-----------------

Command Arguments
~~~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.slurm_container.slurm_container.SlurmContainerCmdArgs
   :members:
   :show-inheritance:

Test Definition
~~~~~~~~~~~~~~~

.. autoclass:: cloudai.workloads.slurm_container.slurm_container.SlurmContainerTestDefinition
   :members:
   :show-inheritance:
