Tutorial
========

This chapter outlines a tutorial on how to utilize the CloudAI framework. Please follow the steps in the same sequence to ensure successful execution:

- :ref:`Step 1: Creating a Docker Image <step-1-creating-docker-image>`
- :ref:`Step 2: Preparing Configuration Files <step-2-preparing-configuration-files>`
- :ref:`Step 3: Testing Definition <step-3-testing-definition>`
- :ref:`Step 4: System Configuration <step-4-system-configuration>`
- :ref:`Step 5: Testing Configuration <step-5-testing-configuration>`
- :ref:`Step 6: Running Experiments <step-6-running-experiments>`
- :ref:`Step 7: Generating Reports <step-7-generating-reports>`
- :ref:`Test in Scenario <test-in-scenario>`

.. _step-1-creating-docker-image:

Creating a Docker Image
~~~~~~~~~~~~~~~~~~~~~~~

To create a Docker image, follow these steps:

#. **Set Up the GitLab Repository:** Start by setting up a repository on GitLab to host your docker image. For this example, use ``gitlab-url.com/cloudai/nccl-test``.

#. **Write the Dockerfile:**

   .. code-block:: dockerfile

      FROM nvcr.io/nvidia/pytorch:24.02-py3

#. **Build and Push the Docker Image:** Build the docker image with the Dockerfile and upload it to the designated repository:

   .. code-block:: bash

      docker build -t gitlab-url.com/cloudai/nccl-test .
      docker push gitlab-url.com/cloudai/nccl-test

#. **Verify the Docker Image:** Test the docker image by running it with ``srun`` to verify that the docker image runs correctly:

   .. code-block:: bash

      srun \
         --mpi=pmix \
         --container-image=gitlab-url.com/cloudai/nccl-test \
         all_reduce_perf_mpi \
         --nthreads 1 \
         --ngpus 1 \
         --minbytes 128 \
         --maxbytes 16G \
         --stepbytes 1M \
         --op sum \
         --datatype float \
         --root 0 \
         --iters 100 \
         --warmup_iters 50 \
         --agg_iters 1 \
         --average 1 \
         --parallel_init 0 \
         --check 1 \
         --blocking 0 \
         --cudagraph 0 \
         --stepfactor 2

.. _step-2-preparing-configuration-files:

Preparing Configuration Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CloudAI is fully configurable via a set of TOML configuration files. You can find examples of these files under ``conf/common``. In this guide, we will use the following configuration files:

#. ``CONFIGS_DIR/system.toml`` - Describes the system configuration.
#. ``CONFIGS_DIR/tests/nccl_test.toml`` - Describes the test to run.
#. ``CONFIGS_DIR/scenario.toml`` - Describes the test scenario configuration.

.. _step-3-testing-definition:

Testing Definition
~~~~~~~~~~~~~~~~~~

Test definition is a Pydantic model that describes the arguments of a test. Such models should be inherited from the ``TestDefinition`` class:

.. code-block:: python

   class MyTestCmdArgs(CmdArgs):
        an_arg: str | list[str]
        docker_image_url: str = "nvcr.io/nvidia/pytorch:24.02-py3"

   class MyTestDefinition(TestDefinition):
       cmd_args: MyTestCmdArgs

Notice that ``cmd_args.docker_image_url`` uses ``nvcr.io/nvidia/pytorch:24.02-py3``, but you can use the Docker image from Step 1.

``an_arg`` has mixed type of ``str | list[str]``, so in a TOML config it can be defined as either:

.. code-block:: toml

   an_arg = "a single string"

Or

.. code-block:: toml

   an_arg = ["list", "of", "strings"]

When a list is used, CloudAI will automatically generate multiple test cases for each value in the list.

A custom test definition should be registered to handle relevant Test Configs. For this, ``Registry()`` object is used:

.. code-block:: python

   Registry().add_test_definition("MyTest", MyTestDefinition)
   Registry().add_test_template("MyTest", MyTest)

Relevant Test Configurations should specify ``test_template_name = MyTest`` to use the custom test definition.

.. _step-4-system-configuration:

System Configuration
~~~~~~~~~~~~~~~~~~~~

System configuration describes how the system configuration works. You can find more examples of system configuration under ``conf/common/system/``. The example below is for demonstration purposes. The following is the ``CONFIGS_DIR/system.toml`` file:

.. code-block:: toml

   name = "my-cluster"
   scheduler = "slurm"

   install_path = "./install"
   output_path = "./results"
   cache_docker_images_locally = true
   default_partition = "<YOUR PARTITION NAME>"

   mpi = "pmix"
   gpus_per_node = 8
   ntasks_per_node = 8

   [[partitions]]
   name = "partition_1"

Replace ``<YOUR PARTITION NAME>`` with the name of the partition you want to use. You can find the partition name by running ``sinfo`` on the cluster.

.. _step-5-testing-configuration:

Testing Configuration
~~~~~~~~~~~~~~~~~~~~~

Test configuration describes a particular test configuration to be run. It is based on test definition and will be used in a test scenario. Below is the ``CONFIGS_DIR/tests/nccl_test.toml`` file, definition is based on the built-in ``NcclTest`` definition:

.. code-block:: toml

   name = "nccl_test_all_reduce_single_node"
   description = "all_reduce"
   test_template_name = "NcclTest"

   [cmd_args]
   subtest_name = "all_reduce_perf_mpi"
   ngpus = 1
   minbytes = "8M"
   maxbytes = "16G"
   iters = 5
   warmup_iters = 3
   stepfactor = 2

You can find more examples under ``conf/common/test``. In a test schema file, you can adjust arguments as shown above. In the ``cmd_args`` section, you can provide different values other than the default values for each argument. In ``extra_cmd_args``, you can provide additional arguments that will be appended after the NCCL test command. You can specify additional environment variables in the ``extra_env_vars`` section.

.. _step-6-running-experiments:

Running Experiments
~~~~~~~~~~~~~~~~~~~

Test Scenario uses test description from Step 5. Below is the ``CONFIGS_DIR/scenario.toml`` file:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "allreduce.1"
   num_nodes = 1
   test_name = "nccl_test_all_reduce_single_node"
   time_limit = "00:20:00"

   [[Tests]]
   id = "allreduce.2"
   num_nodes = 1
   test_name = "nccl_test_all_reduce_single_node"
   time_limit = "00:20:00"
     [[Tests.dependencies]]
     type = "start_post_comp"
     id = "Tests.1"

Notes on the test scenario:

#. ``id`` is a mandatory field and must be unique for each test.
#. The ``test_name`` specifies test definition from one of the Test TOML files. Node lists and time limits are optional.
#. If needed, ``nodes`` should be described as a list of node names as shown in a Slurm system. Alternatively, if groups are defined in the system schema, you can ask CloudAI to allocate a specific number of nodes from a specified partition and group. For example, ``nodes = ['PARTITION:GROUP:16']`` allocates 16 nodes from group ``GROUP`` and partition ``PARTITION``.
#. There are three types of dependencies: ``start_post_comp``, ``start_post_init`` and ``end_post_comp``.

   - ``start_post_comp`` means that the current test should be started after a specific delay of the completion of the depending test.
   - ``start_post_init`` means that the current test should start after the start of the depending test.
   - ``end_post_comp`` means that the current test should be completed after the completion of the depending test.

   All dependencies are described as a pair of the depending test name and a delay. The name should be taken from the test name as set in the test scenario. The delay is described in seconds.

To generate NCCL test commands without actual execution, use the ``dry-run`` mode. You can review ``debug.log`` (or other file specified with ``--log-file``) to see the generated commands from CloudAI. Please note that group node allocations are not currently supported in the ``dry-run`` mode.

.. code-block:: bash

   cloudai dry-run \
       --test-scenario CONFIGS_DIR/scenario.toml \
       --system-config CONFIGS_DIR/system.toml \
       --tests-dir CONFIGS_DIR/tests/

You can run NCCL test experiments with the following command. Whenever you run CloudAI in the ``run`` mode, a new directory will be created under the results directory with the timestamp. In the directory, you can find the results from the test scenario including stdout and stderr. Once completed successfully, you can find generated reports under the directories as well.

.. code-block:: bash

   cloudai run \
       --test-scenario CONFIGS_DIR/scenario.toml \
       --system-config CONFIGS_DIR/system.toml \
       --tests-dir CONFIGS_DIR/tests/


.. _step-7-generating-reports:

Generating Reports
~~~~~~~~~~~~~~~~~~

Once the test scenario is completed, it is possible to generate reports using the following command:

.. code-block:: bash

   cloudai generate-report \
      --test-scenario CONFIGS_DIR/scenario.toml \
      --system-config CONFIGS_DIR/system.toml \
      --tests-dir CONFIGS_DIR/tests/ \
      --result-dir results/2024-06-18_17-40-13/

``--result-dir`` accepts one scenario run result directory.

.. _test-in-scenario:

Test in Scenario
~~~~~~~~~~~~~~~~~~~

It is possible to override some args or even fully define a workload inside a scenario file:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "allreduce.in.scenario"
   num_nodes = 1
   time_limit = "00:20:00"

   name = "nccl_test_all_reduce_single_node"
   description = "all_reduce"
   test_template_name = "NcclTest"

     [Tests.cmd_args]
     subtest_name = "all_reduce_perf_mpi"
     ngpus = 1
     minbytes = "8M"
     maxbytes = "16G"
     iters = 5
     warmup_iters = 3
     stepfactor = 2

   [[Tests]]
   id = "allreduce.override"
   num_nodes = 1
   test_name = "nccl_test_all_reduce_single_node"
   time_limit = "00:20:00"

     [Tests.cmd_args]
     stepfactor = 4

``allreduce.in.scenario`` fully defines a workload; in this case ``test_name`` must not be set, while ``name``, ``description`` and ``test_template_name`` must be set.

``allreduce.override`` overrides only ``stepfactor`` arg from the test defined in the tests directory.

If a scenario contains only fully defined tests, ``--tests-dir`` arg is not required.
