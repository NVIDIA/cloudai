CloudAI User Guide
==================

The purpose of this guide is to help users use CloudAI. The guide covers topics such as adding new tests and downloading datasets for running NeMo-launcher.

Step 1: Create a Docker Image
-----------------------------

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

Step 2: Prepare Configuration Files
-----------------------------------

CloudAI is fully configurable via a set of TOML configuration files. You can find examples of these files under ``conf/common``. In this guide, we will use the following configuration files:

#. ``myconfig/system.toml`` - Describes the system configuration.
#. ``myconfig/tests/nccl_test.toml`` - Describes the test to run.
#. ``myconfig/scenario.toml`` - Describes the test scenario configuration.

Step 3: Test Definition
-----------------------

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

or

.. code-block:: toml

   an_arg = ["list", "of", "strings"]

When a list is used, CloudAI will automatically generate multiple test cases for each value in the list.

A custom test definition should be registered to handle relevant Test Configs. For this, ``Registry()`` object is used:

.. code-block:: python

   Registry().add_test_definition("MyTest", MyTestDefinition)
   Registry().add_test_template("MyTest", MyTest)

Relevant Test Configs should specify ``test_template_name = MyTest`` to use the custom test definition.

Step 4: System Configuration
----------------------------

System configuration describes the system configuration. You can find more examples of system configs under ``conf/common/system/``. Our example will be small for demonstration purposes. Below is the ``myconfig/system.toml`` file:

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

Step 5: Test Configuration
--------------------------

Test configuration describes a particular test configuration to be run. It is based on test definition and will be used in a test scenario. Below is the ``myconfig/tests/nccl_test.toml`` file, definition is based on the built-in ``NcclTest`` definition:

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

Step 6: Run Experiments
-----------------------

Test Scenario uses test description from Step 5. Below is the ``myconfig/scenario.toml`` file:

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
       --test-scenario myconfig/scenario.toml \
       --system-config myconfig/system.toml \
       --tests-dir myconfig/tests/

You can run NCCL test experiments with the following command. Whenever you run CloudAI in the ``run`` mode, a new directory will be created under the results directory with the timestamp. In the directory, you can find the results from the test scenario including stdout and stderr. Once completed successfully, you can find generated reports under the directories as well.

.. code-block:: bash

   cloudai run \
       --test-scenario myconfig/scenario.toml \
       --system-config myconfig/system.toml \
       --tests-dir myconfig/tests/

Test-in-Scenario
----------------

One can override some args or even fully define a workload inside a scenario file:

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

Step 7: Generate Reports
------------------------

Once the test scenario is completed, you can generate reports using the following command:

.. code-block:: bash

   cloudai generate-report \
      --test-scenario myconfig/scenario.toml \
      --system-config myconfig/system.toml \
      --tests-dir myconfig/tests/ \
      --result-dir results/2024-06-18_17-40-13/

``--result-dir`` accepts one scenario run result directory.

Describing a System in the System Schema
----------------------------------------

In this section, we introduce the concept of the system schema, explain the meaning of each field, and describe how the fields should be used. The system schema is a TOML file that allows users to define a system's configuration.

.. code-block:: toml

   name = "example-cluster"
   scheduler = "slurm"

   install_path = "./install"
   output_path = "./results"
   default_partition = "partition_1"

   mpi = "pmix"
   gpus_per_node = 8
   ntasks_per_node = 8

   cache_docker_images_locally = true

   [[partitions]]
   name = "partition_1"

     [[partitions.groups]]
     name = "group_a"
     nodes = ["node-[001-025]"]

     [[partitions.groups]]
     name = "group_b"
     nodes = ["node-[026-050]"]

   [[partitions]]
   name = "partition_2"

   [global_env_vars]
   # NCCL Specific Configurations
   NCCL_IB_GID_INDEX = "3"
   NCCL_IB_TIMEOUT = "20"
   NCCL_IB_QPS_PER_CONNECTION = "4"

   # Device Visibility Configuration
   MELLANOX_VISIBLE_DEVICES = "0,3,4,5,6,9,10,11"
   CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"

Field Descriptions
------------------

- **name**: Specifies the name of the system. Users can choose any name that is convenient for them.
- **scheduler**: Indicates the type of system. It should be one of the supported types, currently ``slurm`` or ``standalone``. ``slurm`` refers to a system with the Slurm scheduler, while ``standalone`` refers to a single-node system without any slave nodes. Other values are possible depending on the available schedulers supported by CloudAI.
- **install_path**: Specifies the path where test prerequisites are installed. Docker images are downloaded to this path if the user chooses to cache Docker images.
- **output_path**: Defines the default path where outputs are stored. Whenever a user runs a test scenario, a new subdirectory will be created under this path.
- **default_partition**: Specifies the default partition where jobs are scheduled.
- **partitions**: Describes the available partitions and nodes within those partitions.

  - **[optional] groups**: Within the same partition, users can define groups of nodes. The group concept can be used to allocate nodes from specific groups in a test scenario schema. For instance, this feature is useful for specifying topology awareness. Groups represent logical partitioning of nodes and users are responsible for ensuring no overlap across groups.

- **mpi**: Indicates the Process Management Interface (PMI) implementation to be used for inter-process communication.
- **gpus_per_node** and **ntasks_per_node**: These are Slurm arguments passed to the ``sbatch`` script and ``srun``.
- **cache_docker_images_locally**: Specifies whether CloudAI should cache remote Docker images locally during installation. If set to ``true``, CloudAI will cache the Docker images, enabling local access without needing to download them each time a test is run. This approach saves network bandwidth but requires more disk capacity. If set to ``false``, CloudAI will allow Slurm to download the Docker images as needed when they are not cached locally by Slurm.
- **global_env_vars**: Lists all global environment variables that will be applied globally whenever tests are run.

Describing a System for RunAI Scheduler
---------------------------------------

When using RunAI as the scheduler, you need to specify additional fields in the system schema TOML file. Below is the list of required fields and how to set them:

.. code-block:: toml

   name = "runai-cluster"
   scheduler = "runai"

   install_path = "./install"
   output_path = "./results"

   base_url = "http://runai.example.com"       # The URL of your RunAI system, typically the same as used for the web interface.
   user_email = "your_email"              # The email address used to log into the RunAI system.
   app_id = "your_app_id"                      # Obtained by creating an application in the RunAI web interface.
   app_secret = "your_app_secret"              # Obtained together with the app_id.
   project_id = "your_project_id"              # Project ID assigned or created in the RunAI system (usually an integer).
   cluster_id = "your_cluster_id"              # Cluster ID in UUID format (e.g., a69928cc-ccaa-48be-bda9-482440f4d855).

* After logging into the RunAI web interface, navigate to Access → Applications and create a new application to obtain ``app_id`` and ``app_secret``.
* Use your assigned project and cluster IDs. Contact your administrator if they are not available.
* All other fields follow the same semantics as in the Slurm system schema (e.g., ``install_path``, ``output_path``).

Describing a Test Scenario in the Test Scenario Schema
------------------------------------------------------

A test scenario is a set of tests with specific dependencies between them. A test scenario is described in a TOML schema file. This is an example of a test scenario file:

.. code-block:: toml

   name = "nccl-test"

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   num_nodes = "2"
   time_limit = "00:20:00"

   [[Tests]]
   id = "Tests.2"
   test_name = "nccl_test_all_gather"
   num_nodes = "2"
   time_limit = "00:20:00"
     [[Tests.dependencies]]
     type = "start_post_comp"
     id = "Tests.1"

   [[Tests]]
   id = "Tests.3"
   test_name = "nccl_test_reduce_scatter"
   num_nodes = "2"
   time_limit = "00:20:00"
     [[Tests.dependencies]]
     type = "start_post_comp"
     id = "Tests.2"

The ``name`` field is the test scenario name, which can be any unique identifier for the scenario. Each test has a section name, following the convention ``Tests.1``, ``Tests.2``, etc., with an increasing index. The ``name`` of a test should be specified in this section and must correspond to an entry in the test schema. If a test in a test scenario is not present in the test schema, CloudAI will not be able to identify it.

There are two ways to specify nodes:

- Using the ``num_nodes`` field as shown in the example.
- Specifying nodes explicitly like ``nodes = ["node-001", "node-002"]``

**Note:** When an explicit node list is provided (e.g., ``nodes = ["node-001", "node-002"]``), CloudAI lets Slurm apply the arbitrary distribution policy for task placement.

Alternatively, you can utilize the groups feature in the system schema to specify nodes like ``nodes = ['PARTITION_NAME:GROUP_NAME:NUM_NODES']``, which allocates ``num_nodes`` from the group name in the specified partition. You can also use ``nodes = ['PARTITION_NAME:GROUP_NAME:max_avail']``, which allocates all the available nodes from the group name in the specified partition.

You can optionally specify a time limit in the Slurm format. Tests can have dependencies. If no dependencies are specified, all tests will run in parallel.

CloudAI supports three types of dependencies:

- ``start_post_init``
- ``start_post_comp``
- ``end_post_comp``

Dependencies of a test can be described as a subsection of the test. It requires other tests' ``id`` and dependency ``type``.

- ``start_post_init`` means the test starts after the prior test begins, with a specified delay.
- ``start_post_comp`` means the test starts after the prior test completes.
- ``end_post_comp`` means the test ends when the prior test completes.

Configuring HTTP Data Repository
--------------------------------

The HTTP Data Repository is currently supported for Slurm systems only. To enable access, you must update your system schema file and create a credential file in your CloudAI project's root directory.

Step 1: Update the System Schema File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the following section to your system schema TOML file (e.g., ``system_schema.toml``):

.. code-block:: toml

   [data_repository]
   endpoint = "https://my-data-endpoint.com"

Replace the endpoint with your actual data repository URL.

Step 2: Create the Credential File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the root of your CloudAI project (i.e., the current working directory), create a file named ``.cloudai.toml`` with the following content:

.. code-block:: toml

   [data_repository]
   token = "<your-api-token-here>"

Replace ``<your-api-token-here>`` with your actual token.

Step 3: Usage
~~~~~~~~~~~~~

Both the endpoint and token must be valid for the HTTP Data Repository to function correctly. If either is missing or incorrect, data will not be posted.

Using Test Hooks in CloudAI
---------------------------

A test hook in CloudAI is a specialized test that runs either before or after each main test in a scenario, providing flexibility to prepare the environment or clean up resources. Hooks are defined as pre-test or post-test and referenced in the test scenario’s TOML file using ``pre_test`` and ``post_test`` fields.

.. code-block:: toml

   name = "nccl-test"

   pre_test = "nccl_test_pre"
   post_test = "nccl_test_post"

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   time_limit = "00:20:00"

CloudAI organizes hooks in a dedicated directory structure:

- Hook directory: All hook configurations reside in ``conf/hook/``.
- Hook test scenarios: Place pre-test hook and post-test hook scenario files in ``conf/hook/``.
- Hook tests: Place individual tests referenced in hooks within ``conf/hook/test/``.

In the execution flow, pre-test hooks run before the main test, which only executes if the pre-test completes successfully. Post-test hooks follow the main test, provided the prior steps succeed. If a pre-test hook fails, the main test and its post-test hook are skipped.

.. code-block:: toml

   name = "nccl_test_pre"

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   time_limit = "00:20:00"

Downloading DeepSeek Weights
----------------------------

To run DeepSeek R1 tests in CloudAI, you must download the model weights in advance. These weights are distributed via the NVIDIA NGC Registry and must be manually downloaded using the NGC CLI.

Step 1: Install NGC CLI
~~~~~~~~~~~~~~~~~~~~~~~

Download and install the NGC CLI using the following commands:

.. code-block:: bash

   wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.2/files/ngccli_linux.zip -O ngccli_linux.zip
   unzip ngccli_linux.zip
   chmod u+x ngc-cli/ngc
   echo "export PATH=\"$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile

This will make the ``ngc`` command available in your terminal.

Step 2: Configure NGC CLI
~~~~~~~~~~~~~~~~~~~~~~~~~

Authenticate your CLI with your NGC API key by running:

.. code-block:: bash

   ngc config set

When prompted, paste your API key, which you can obtain from https://org.ngc.nvidia.com/setup.

Step 3: Download the Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to the directory where you want the DeepSeek model weights to be stored, then run:

.. code-block:: bash

   ngc registry model download-version nim/deepseek-ai/deepseek-r1-instruct:hf-5dde110-nim-fp8 --dest .

This command will create a folder named:

.. code-block:: text

   deepseek-r1-instruct_vhf-5dde110-nim-fp8/

inside your current directory.

Step 4: Verify the Download
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure the full model has been downloaded by checking the folder size:

.. code-block:: bash

   du -sh deepseek-r1-instruct_vhf-5dde110-nim-fp8

The expected size is approximately 642 GB. If it’s significantly smaller, remove the folder and re-run the download.

Slurm specifics
---------------

Single sbatch vs per-case sbatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CloudAI supports two modes for submitting Slurm jobs:

1. (default) Per-case sbatch mode: each case is submitted as a separate sbatch job, allowing flexible scheduling and dependency management.

   - Suits best when cases can run in parallel or there are no dependencies between cases.

2. Single sbatch mode: all cases are submitted together in a single sbatch job and share the same nodes. Each next case starts after the previous one completes. To enable it one needs to pass ``--single-sbatch`` flag to ``cloudai run`` command (works only for Slurm systems).

   - Suits best for jobs when cases need to run on the same nodes for performance reasons.
   - There is no support for dependency management between cases yet; all jobs run one after another.

Assuming two cases in a scenario like this:

.. code-block:: toml

   [[Tests]]
   id = "nccl.all_reduce"
   test_name = "nccl-all_reduce"
   num_nodes = 2
   time_limit = "00:20:00"

   [[Tests]]
   id = "nccl.all_gather"
   test_name = "nccl-all_gather"
   num_nodes = 2
   time_limit = "00:20:00"

Regular output directory structure (some files are omitted for clarity):

.. code-block:: bash

   $ tree results/scenario
   results/scenario
   ├── nccl.all_gather
   │   └── 0
   │       ├── cloudai_sbatch_script.sh
   │       ├── env_vars.sh
   │       ├── metadata/
   │       ├── stderr.txt
   │       ├── stdout.txt
   │       └── test-run.toml
   └── nccl.all_reduce
       └── 0
           ├── cloudai_sbatch_script.sh
           ├── env_vars.sh
           ├── metadata/
           ├── stderr.txt
           ├── stdout.txt
           └── test-run.toml

Single sbatch mode output directory structure:

.. code-block:: bash

   $ tree results/scenario
   results/scenario
   ├── cloudai_sbatch_script.sh
   ├── common.err
   ├── common.out
   ├── metadata/
   ├── nccl.all_gather
   │   └── 0
   │       ├── env_vars.sh
   │       ├── stderr.txt
   │       ├── stdout.txt
   │       └── test-run.toml
   ├── nccl.all_reduce
   │   └── 0
   │       ├── env_vars.sh
   │       ├── stderr.txt
   │       ├── stdout.txt
   │       └── test-run.toml
   └── slurm-job.toml

Most of the files are the same: output files, env vars script, test run metadata. The difference for single sbatch mode is that ``slurm-job.toml`` is created for the entire job as well as ``cloudai_sbatch_script.sh``. Extra ``common.err``/``common.out`` files are created for general sbatch outputs.

Extra srun and sbatch arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CloudAI forms sbatch script and srun commands following internal rules. Users can affect the generation by setting special arguments in the system TOML file.

For example (in a system TOML file):

.. code-block:: toml

   extra_sbatch_args = [
     "--section=4",
     "--other-arg val"
   ]

will result in sbatch file content like this:

.. code-block:: bash

   ... # CloudAI set sbatch arguments
   #SBATCH --section=4
   #SBATCH --other-arg val
   ...

Another example (in a system TOML file):

.. code-block:: toml

   extra_srun_args = "--arg=val --other-arg=other-val"

will result in srun command inside sbatch script like this:

.. code-block:: bash

   srun ... --arg=val --other-arg=other-val ...

Container mounts
~~~~~~~~~~~~~~~~

CloudAI runs all slurm jobs using containers. To simplify file system related tasks, CloudAI mounts the following directories into the container:

1. Test output directory (``<output_path>/<scenario_name_with_timestamp>/<test_name>/<iteration>``, like ``results/scenario_2024-06-18_17-40-13/Tests.1/0``) is mounted as ``/cloudai_run_results``.
2. Test specific mounts can be specified in TOML files:

   .. code-block:: toml

      extra_container_mounts = [
        "/path/to/mount1:/path/in/container1",
        "/path/to/mount2:/path/in/container2"
      ]

      [cmd_args]
      ...

   These mounts are not verified for validity and do not override default mounts.

3. Test specific mounts can be mounted in-code.

Head node without shared storage available on compute nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When compute nodes don't share file system with head node, ``--enable-cache-without-check`` for ``run`` and ``dry-run`` skips the real check for cache existence, but still builds all paths correctly. The flow is like this:

1. *[on the head node]* run ``cloudai install``.
2. *[on the head node]* copy cache to compute nodes.
3. Modify ``system.toml`` to set compute nodes' installation root.
4. Run ``cloudai run --enable-cache-without-check ...``.

Dev details
-----------

``SlurmCommandGenStrategy`` defines abstract method ``_container_mounts(tr: TestRun)`` that must be implemented by every subclass. This method is used in ``SlurmCommandGenStrategy.container_mounts(tr: TestRun)`` (defined as ``@final``) where mounts like ``/cloudai_run_results`` (default mount), ``TestDefinition.extra_container_mounts`` (from Test TOML) and test specific mounts (defined in-code) are added.

Nsys tracing
------------

Users can enable Nsys tracing for any workload when running via Slurm. Note that ``nsys`` should be available on the compute nodes; CloudAI doesn't manage it.

Configuration fields are:

.. code-block:: python

   enable: bool = True
   nsys_binary: str = "nsys"
   task: str = "profile"
   output: Optional[str] = None
   sample: Optional[str] = None
   trace: Optional[str] = None
   force_overwrite: Optional[bool] = None
   capture_range: Optional[str] = None
   capture_range_end: Optional[str] = None
   cuda_graph_trace: Optional[str] = None
   gpu_metrics_devices: Optional[str] = None
   extra_args: list[str] = []

Fields with ``None`` value are not passed to ``nsys`` command.

Troubleshooting
---------------

In this section, we will guide you through identifying the root cause of issues, determining whether they stem from system infrastructure or a bug in CloudAI.

Identifying the Root Cause
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues running a command, start by reading the error message to understand the root cause. We strive to make our error messages and exception messages as readable and interpretable as possible.

System Infrastructure vs. CloudAI Bugs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To determine whether an issue is due to system infrastructure or a CloudAI bug, follow these steps:

1. **Check stdout Messages:** If CloudAI fails to run a test successfully, it will be indicated in the stdout messages that a test has failed.

2. **Review Log Files:**

   - Navigate to the output directory and review ``debug.log``, stdout, and stderr files.
   - ``debug.log`` contains detailed steps executed by CloudAI, including generated commands, executed commands, and error messages.

3. **Analyze Error Messages:** By examining the error messages in the log files, you can understand the type of errors CloudAI encountered.

4. **Examine Output Directory:** If a test fails without explicit error messages, review the output directory of the failed test. Look for ``stdout.txt``, ``stderr.txt``, or any generated files to understand the failure reason.

5. **Manual Rerun of Tests:**

   - To manually rerun the test, consult the ``debug.log`` for the command CloudAI executed.
   - Look for an ``sbatch`` command with a generated ``sbatch`` script.
   - Execute the command manually to debug further.

If the problem persists, please report the issue at https://github.com/NVIDIA/cloudai/issues/new/choose. When you report an issue, ensure it is reproducible. Follow the issue template and provide any necessary details, such as the hash commit used, system settings, any changes in the schema files, and the command.

