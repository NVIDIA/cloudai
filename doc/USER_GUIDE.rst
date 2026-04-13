User Guide
==========

The purpose of this chapter is to help users utilize and understand the different components of the CloudAI framework. The chapter covers topics such as adding new tests and downloading datasets for running NeMo-launcher.

System Schema
--------------

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

.. list-table:: Field Descriptions
   :header-rows: 1
   :widths: 25 75

   * - **Field**
     - **Description**
   * - **name**
     - Specifies the name of the system. Users can choose any name that is convenient for them.
   * - **scheduler**
     - Indicates the type of system. It should be one of the supported types, currently ``slurm`` or ``standalone``. ``slurm`` refers to a system with the Slurm scheduler, while ``standalone`` refers to a single-node system without any slave nodes. Other values are possible depending on the available schedulers supported by CloudAI.
   * - **install_path**
     - Specifies the path where test prerequisites are installed. Docker images are downloaded to this path if the user chooses to cache Docker images.
   * - **output_path**
     - Defines the default path where outputs are stored. Whenever a user runs a test scenario, a new subdirectory will be created under this path.
   * - **default_partition**
     - Specifies the default partition where jobs are scheduled.
   * - **partitions**
     - Describes the available partitions and nodes within those partitions.
   * - **[optional] groups**
     - Within the same partition, users can define groups of nodes. The group concept can be used to allocate nodes from specific groups in a test scenario schema. For instance, this feature is useful for specifying topology awareness. Groups represent logical partitioning of nodes and users are responsible for ensuring no overlap across groups.
   * - **mpi**
     - Indicates the Process Management Interface (PMI) implementation to be used for inter-process communication.
   * - **gpus_per_node** and **ntasks_per_node**
     - These are Slurm arguments passed to the ``sbatch`` script and ``srun``.
   * - **cache_docker_images_locally**
     - Specifies whether CloudAI should cache remote Docker images locally during installation. If set to ``true``, CloudAI will cache the Docker images, enabling local access without needing to download them each time a test is run. This approach saves network bandwidth but requires more disk capacity. If set to ``false``, CloudAI will allow Slurm to download the Docker images as needed when they are not cached locally by Slurm.
   * - **global_env_vars**
     - Lists all global environment variables that will be applied globally whenever tests are run.

RunAI Scheduler
---------------

When using RunAI as the scheduler, additional fields must be specified in the system schema TOML file. The following is a list of required fields and how to set them:

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

- After logging into the RunAI web interface, navigate to Access → Applications and create a new application to obtain ``app_id`` and ``app_secret``
- Use your assigned project and cluster IDs. Contact your administrator if they are not available
- All other fields follow the same semantics as in the Slurm system schema (e.g., ``install_path``, ``output_path``)

Test Scenario Schema
--------------------

A test scenario is a set of tests with specific dependencies between them. A test scenario is described in a TOML schema file. The following is an example of a test scenario file:

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

- Using the ``num_nodes`` field as shown in the example
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

Agents Configuration
--------------------

For DSE workloads, you can pass agent configuration via ``agent_config`` in the scenario.

``BaseAgentConfig`` includes:

- ``random_seed``: controls deterministic random behavior in agents.
- ``start_action``: controls the very first action strategy (``"random"`` or ``"first"``).
- ``rewards`` (optional): nested table mapped to ``RewardOverrides``. When present, ``CloudAIGymEnv`` uses it for
  constraint failures and for substituting failed-metric values in observations.

  - ``constraint_failure``: reward returned when ``TestDefinition.constraint_check`` fails on a step. If omitted
    or unset, the environment uses ``-1.0``.
  - ``metric_failure``: value written into the observation vector when a metric is missing or is the
    canonical error sentinel ``METRIC_ERROR`` (a distinct object from numeric metrics, so a valid ``-1.0``
    result is not mistaken for an error). If omitted or unset, observations use ``-1.0`` for that slot.

Example:

.. code-block:: toml

   [[Tests]]
   id = "Tests.1"
   test_name = "nccl_test_all_reduce"
   agent = "grid_search"
   agent_steps = 10
   agent_metrics = ["default"]
   agent_reward_function = "inverse"

     [Tests.agent_config]
     random_seed = 123
     start_action = "first"

       [Tests.agent_config.rewards]
       constraint_failure = -5.0
       metric_failure = 0.0

When an agent honors ``start_action = "first"``, it should start from ``CloudAIGymEnv.first_sweep`` (the sweep
is built from first values of each sweep parameter). ``start_action = "random"`` means starting from a random
action, typically seeded by ``random_seed``.

Custom agents may extend the ``BaseAgentConfig`` and offer more parameters to configure.

Configuring HTTP Data Repository
--------------------------------

The HTTP Data Repository is currently supported for Slurm systems only. To enable access, you must update your system schema file and create a credential file in your CloudAI project's root directory.

The following steps are required to configure the HTTP Data Repository:

- :ref:`Step 1: Update the System Schema File <step-1-update-system-schema-file>`
- :ref:`Step 2: Create the Credential File <step-2-create-credential-file>`
- :ref:`Step 3: Usage <step-3-usage>`

.. _step-1-update-system-schema-file:

Updating the System Schema File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the following section to your system schema TOML file (e.g., ``system_schema.toml``):

.. code-block:: toml

   [data_repository]
   endpoint = "https://my-data-endpoint.com"

Replace the endpoint with your actual data repository URL.

.. _step-2-create-credential-file:

Creating the Credential File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the root of your CloudAI project (i.e., the current working directory), create a file named ``.cloudai.toml`` with the following content:

.. code-block:: toml

   [data_repository]
   token = "<your-api-token-here>"

Replace ``<your-api-token-here>`` with your actual token.

.. _step-3-usage:

Usage
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

- Hook directory: All hook configurations reside in ``conf/hook/``
- Hook test scenarios: Place pre-test hook and post-test hook scenario files in ``conf/hook/``
- Hook tests: Place individual tests referenced in hooks within ``conf/hook/test/``

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

The following steps are required to download the DeepSeek weights:

- :ref:`Step 1: Installing NGC CLI <step-1-installing-ngc-cli>`
- :ref:`Step 2: Configuring NGC CLI <step-2-configuring-ngc-cli>`
- :ref:`Step 3: Downloading the Weights <step-3-downloading-the-weights>`
- :ref:`Step 4: Verifying the Download <step-4-verifying-the-download>`

.. _step-1-installing-ngc-cli:

Installing NGC CLI
~~~~~~~~~~~~~~~~~~~

Download and install the NGC CLI using the following commands:

.. code-block:: bash

   wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.2/files/ngccli_linux.zip -O ngccli_linux.zip
   unzip ngccli_linux.zip
   chmod u+x ngc-cli/ngc
   echo "export PATH=\"$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile

This will make the ``ngc`` command available in your terminal.

.. _step-2-configuring-ngc-cli:

Configuring NGC CLI
~~~~~~~~~~~~~~~~~~~

Authenticate your CLI with your NGC API key by running:

.. code-block:: bash

   ngc config set

When prompted, paste your API key, which you can obtain from https://org.ngc.nvidia.com/setup.

.. _step-3-downloading-the-weights:

Downloading the Weights
~~~~~~~~~~~~~~~~~~~~~~~~

Navigate to the directory where you want the DeepSeek model weights to be stored, then run:

.. code-block:: bash

   ngc registry model download-version nim/deepseek-ai/deepseek-r1-instruct:hf-5dde110-nim-fp8 --dest .

This command will create a folder named:

.. code-block:: text

   deepseek-r1-instruct_vhf-5dde110-nim-fp8/

inside your current directory.

.. _step-4-verifying-the-download:

Verifying the Download
~~~~~~~~~~~~~~~~~~~~~~~

Make sure the full model has been downloaded by checking the folder size:

.. code-block:: bash

   du -sh deepseek-r1-instruct_vhf-5dde110-nim-fp8

The expected size is approximately 642 GB. If it’s significantly smaller, remove the folder and re-run the download.

Slurm
-----

The following are subtopics related to Slurm:

Single sbatch vs Per-case sbatch
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

Extra srun and sbatch Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CloudAI forms sbatch script and srun commands following internal rules. Users can affect the generation by setting special arguments in the system TOML file.

For example (in a system TOML file):

.. code-block:: toml

   extra_sbatch_args = [
     "--section=4",
     "--other-arg val"
   ]

It will result in sbatch file content like this:

.. code-block:: bash

   ... # CloudAI set sbatch arguments
   #SBATCH --section=4
   #SBATCH --other-arg val
   ...

Another example (in a system TOML file):

.. code-block:: toml

   extra_srun_args = "--arg=val --other-arg=other-val"

Will result in srun command inside sbatch script like this:

.. code-block:: bash

   srun ... --arg=val --other-arg=other-val ...

Container Mounts
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

Head Node without Shared Storage Available on Compute Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When compute nodes do not share the file system with head node, ``--enable-cache-without-check`` for ``run`` and ``dry-run`` skips the real check for cache existence, but still builds all paths correctly. The flow is like this:

1. *[on the head node]* run ``cloudai install``.
2. *[on the head node]* copy cache to compute nodes.
3. Modify ``system.toml`` to set compute nodes' installation root.
4. Run ``cloudai run --enable-cache-without-check ...``.

Dev Details
~~~~~~~~~~~

``SlurmCommandGenStrategy`` defines abstract method ``_container_mounts(tr: TestRun)`` that must be implemented by every subclass. This method is used in ``SlurmCommandGenStrategy.container_mounts(tr: TestRun)`` (defined as ``@final``) where mounts like ``/cloudai_run_results`` (default mount), ``TestDefinition.extra_container_mounts`` (from Test TOML) and test specific mounts (defined in-code) are added.

Nsys Tracing
~~~~~~~~~~~~

Users can enable Nsys tracing for any workload when running via Slurm. Note that ``nsys`` should be available on the compute nodes; CloudAI does not manage it.

The configuration fields are:

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
