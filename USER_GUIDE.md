# CloudAI User Guide
This is a CloudAI user guide to help users use CloudAI, covering topics such as adding new tests and downloading datasets for running NeMo-launcher.

#### Step 1: Create a Docker Image
1. **Set Up the GitLab Repository:**
  Start by setting up a repository on GitLab to host your docker image. For this example, use `gitlab-url.com/cloudai/nccl-test`.

2. **Write the Dockerfile:**
   The Dockerfile needs to specify the base image and detail the steps:
   ```dockerfile
   FROM nvcr.io/nvidia/pytorch:24.02-py3
   ```

3. **Build and Push the Docker Image:**
   Build the docker image with the Dockerfile and upload it to the designated repository:
   ```bash
   docker build -t gitlab-url.com/cloudai/nccl-test .
   docker push gitlab-url.com/cloudai/nccl-test
   ```

4. **Verify the Docker Image:**
   Test the docker image by running it with `srun` to verify that the docker image runs correctly:
   ```bash
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
   ```

#### Step 2: Prepare Configuration Files
CloudAI is fully configurable via set of TOML configuration files. You can find examples of these files under `conf/common`. In this guide, we will use the following configuration files:
1. `myconfig/system.toml` - Describes the system configuration.
1. `myconfig/tests/nccl_test.toml` - Describes the test to run.
1. `myconfig/scenario.toml` - Describes the test scenario configuration.


#### Step 3: Test Definition
Test definition is a Pydantic model that describes the arguments of a test. Such models should be inherited from the `TestDefinition` class:
```py
class MyTestCmdArgs(CmdArgs):
     an_arg: str
     docker_image_url: str = "nvcr.io/nvidia/pytorch:24.02-py3"

class MyTestDefinition(TestDefinition):
    cmd_args: MyTestCmdArgs
```
Notice that `cmd_args.docker_image_url` uses `nvcr.io/nvidia/pytorch:24.02-py3`, but you can use Docker image from Step 1.

A custom test definition should be registered to handle relevant Test Configs. For this, `Registry()` object is used:
```py
Registry().add_test_definition("MyTest", MyTestDefinition)
Registry().add_test_template("MyTest", MyTest)
```
Relevant Test Configs should specify `test_template_name = MyTest` to use the custom test definition.

#### Step 3: System Configuration
System configuration describes the system configuration. You can find more examples of system configs under `conf/common/system/`. Our example will be small for demonstration purposes. Below is the `myconfig/system.toml` file:
```toml
name = "my-cluster"
scheduler = "slurm"

install_path = "./install"
output_path = "./results"
cache_docker_images_locally = "True"
default_partition = "<YOUR PARTITION NAME>"

mpi = "pmix"
gpus_per_node = 8
ntasks_per_node = 8

[partitions]
  [partitions.<YOUR PARTITION NAME>]
  name = "<YOUR PARTITION NAME>"
```
Replace `<YOUR PARTITION NAME>` with the name of the partition you want to use. You can find the partition name by running `sinfo` on the cluster.

#### Step 4: Install Test Requirements
Once all configs are ready, it is time to install test requirements. It is done once so that you can run multiple experiments without reinstalling the requirements. This step requires the system config file from the step 3.
```bash
cloudai install \
   --system-config myconfig/system.toml \
   --tests-dir myconfig/tests/
```

#### Step 5: Test Configuration
Test Configuration describes a particular test configuration to be run. It is based on Test definition and will be used in Test Sceanrio. Below is the `myconfig/tests/nccl_test.toml` file, definition is based on built-in `NcclTest` definition:
```toml
name = "nccl_test_all_reduce_single_node"
description = "all_reduce"
test_template_name = "NcclTest"
extra_cmd_args = "--stepfactor 2"

[cmd_args]
"subtest_name" = "all_reduce_perf_mpi"
"ngpus" = "1"
"minbytes" = "8M"
"maxbytes" = "16G"
"iters" = "5"
"warmup_iters" = "3"
```
You can find more examples under `conf/common/test`. In a test schema file, you can adjust arguments as shown above. In the `cmd_args` section, you can provide different values other than the default values for each argument. In `extra_cmd_args`, you can provide additional arguments that will be appended after the NCCL test command. You can specify additional environment variables in the `extra_env_vars` section.

#### Step 6: Run Experiments
Test Scenario uses Test description from step 5. Below is the `myconfig/scenario.toml` file:
```toml
name = "nccl-test"

[[Tests]]
id = "Tests.1"
test_name = "nccl_test_all_reduce_single_node"
time_limit = "00:20:00"

[[Tests]]
id = "Tests.2"
test_name = "nccl_test_all_reduce_single_node"
time_limit = "00:20:00"
  [[Tests.dependencies]]
  type = "start_post_comp"
  id = "Tests.1"
```

Notes on the test scenario:
1. `id` is a mandatory filed and must be unique for each test.
1. The `test_name` specifies test definition from one of the Test TOML files. Node lists and time limits are optional.
1. If needed, `nodes` should be described as a list of node names as shown in a Slurm system. Alternatively, if groups are defined in the system schema, you can ask CloudAI to allocate a specific number of nodes from a specified partition and group. For example, `nodes = ['PARTITION:GROUP:16']`: 16 nodes are allocated from a group `GROUP`, from a partition `PARTITION`.
1. There are three types of dependencies: `start_post_comp`, `start_post_init` and `end_post_comp`.
    1. `start_post_comp` means that the current test should be started after a specific delay of the completion of the depending test.
    1. `start_post_init` means that the current test should start after the start of the depending test.
    1. `end_post_comp` means that the current test should be completed after the completion of the depending test.

   All dependencies are described as a pair of the depending test name and a delay. The name should be taken from the test name as set in the test scenario. The delay is described in the number of seconds.


To generate NCCL test commands without actual execution, use the `dry-run` mode. You can review `debug.log` (or other file specifued with `--log-file`) to see the generated commands from CloudAI. Please note that group node allocations are not currently supported in the `dry-run` mode.
```bash
cloudai dry-run \
    --test-scenario myconfig/scenario.toml \
    --system-config myconfig/system.toml \
    --tests-dir myconfig/tests/
```

You can run NCCL test experiments with the following command. Whenever you run CloudAI in the `run` mode, a new directory will be created under the results directory with the timestamp. In the directory, you can find the results from the test scenario including stdout and stderr. Once completed successfully, you can find generated reports under the directories as well.
```bash
cloudai run \
    --test-scenario myconfig/scenario.toml \
    --system-config myconfig/system.toml \
    --tests-dir myconfig/tests/
```

#### Step 7: Generate Reports
Once the test scenario is completed, you can generate reports using the following command:
```bash
cloudai generate-report \
   --test-scenario myconfig/scenario.toml \
   --system-config myconfig/system.toml \
   --tests-dir myconfig/tests/ \
   --result-dir results/2024-06-18_17-40-13/
```

`--result-dir` accepts one scenario run result directory.

## Describing a System in the System Schema
In this section, we introduce the concept of the system schema, explain the meaning of each field, and describe how the fields should be used. The system schema is a TOML file that allows users to define a system's configuration.

```
name = "example-cluster"
scheduler = "slurm"

install_path = "./install"
output_path = "./results"
default_partition = "partition_1"

mpi = "pmix"
gpus_per_node = 8
ntasks_per_node = 8

cache_docker_images_locally = true

[partitions]
  [partitions.partition_1]
  name = "partition_1"
  nodes = ["node-[001-100]"]

  [partitions.partition_2]
  name = "partition_2"
  nodes = ["node-[101-200]"]

  [partitions.partition_1.groups]
    [partitions.partition_1.groups.group_1]
    name = "group_1"
    nodes = ["node-[001-025]"]

    [partitions.partition_1.groups.group_2]
    name = "group_2"
    nodes = ["node-[026-050]"]

    [partitions.partition_1.groups.group_3]
    name = "group_3"
    nodes = ["node-[051-075]"]

    [partitions.partition_1.groups.group_4]
    name = "group_4"
    nodes = ["node-[076-100]"]

[global_env_vars]
  # NCCL Specific Configurations
  NCCL_IB_GID_INDEX = "3"
  NCCL_IB_TIMEOUT = "20"
  NCCL_IB_QPS_PER_CONNECTION = "4"

  # Device Visibility Configuration
  MELLANOX_VISIBLE_DEVICES = "0,3,4,5,6,9,10,11"
  CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
```

### Field Descriptions
- **name**: Specifies the name of the system. Users can choose any name that is convenient for them.
- **scheduler**: Indicates the type of system. It should be one of the supported types, currently `slurm` or `standalone`. `slurm` refers to a system with the Slurm scheduler, while `standalone` refers to a single-node system without any slave nodes.
- **install_path**: Specifies the path where test prerequisites are installed. Docker images are downloaded to this path if the user chooses to cache Docker images.
- **output_path**: Defines the default path where outputs are stored. Whenever a user runs a test scenario, a new subdirectory will be created under this path.
- **default_partition**: Specifies the default partition where jobs are scheduled.
- **partitions**: Describes the available partitions and nodes within those partitions.
  - **[optional] groups**: Within the same partition, users can define groups of nodes. The group concept can be used to allocate nodes from specific groups in a test scenario schema. For instance, this feature is useful for specifying topology awareness. Groups represents logical partitioning of nodes and users are responsible for ensuring no overlap across groups.
- **mpi**: Indicates the Process Management Interface (PMI) implementation to be used for inter-process communication.
- **gpus_per_node** and **ntasks_per_node**: These are Slurm arguments passed to the `sbatch` script and `srun`.
- **cache_docker_images_locally**: Specifies whether CloudAI should cache remote Docker images locally during installation. If set to `true`, CloudAI will cache the Docker images, enabling local access without needing to download them each time a test is run. This approach saves network bandwidth but requires more disk capacity. If set to `false`, CloudAI will allow Slurm to download the Docker images as needed when they are not cached locally by Slurm.
- **global_env_vars**: Lists all global environment variables that will be applied globally whenever tests are run.

## Describing a System for RunAI Scheduler
When using RunAI as the scheduler, you need to specify additional fields in the system schema TOML file. Below is the list of required fields and how to set them:

```toml
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
```
* After logging into the RunAI web interface, navigate to Access → Applications and create a new application to obtain app_id and app_secret.
* Use your assigned project and cluster IDs. Contact your administrator if they are not available.
* All other fields follow the same semantics as in the Slurm system schema (e.g., install_path, output_path).

## Describing a Test Scenario in the Test Scenario Schema
A test scenario is a set of tests with specific dependencies between them. A test scenario is described in a TOML schema file. This is an example of a test scenario file:
```toml
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
templat_test = "nccl_test_reduce_scatter"
num_nodes = "2"
time_limit = "00:20:00"
  [[Tests.dependencies]]
  type = "start_post_comp"
  id = "Tests.2"
```

The `name` field is the test scenario name, which can be any unique identifier for the scenario. Each test has a section name, following the convention `Tests.1`, `Tests.2`, etc., with an increasing index. The `name` of a test should be specified in this section and must correspond to an entry in the test schema. If a test in a test scenario is not present in the test schema, CloudAI will not be able to identify it.

There are two ways to specify nodes:
-  Using the `num_nodes` field as shown in the example.
-  Specifying nodes explicitly like `nodes = ["node-001", "node-002"]`

**Note:** When an explicit node list is provided (e.g., `nodes = ["node-001", "node-002"]`), CloudAI lets Slurm apply the arbitrary distribution policy for task placement.

 Alternatively, you can utilize the groups feature in the system schema to specify nodes like `nodes = ['PARTITION_NAME:GROUP_NAME:NUM_NODES']`, which allocates `num_nodes` from the group name in the specified partition. You can also use `nodes = ['PARTITION_NAME:GROUP_NAME:max_avail']`, which allocates all the available nodes from the group name in the specified partition.

You can optionally specify a time limit in the Slurm format. Tests can have dependencies. If no dependencies are specified, all tests will run in parallel.

CloudAI supports three types of dependencies:
- `start_post_init`
- `start_post_comp`
- `end_post_comp`

Dependencies of a test can be described as a subsection of the test. It requires other tests' `id` and dependency `type`.

- `start_post_init` means the test starts after the prior test begins, with a specified delay
- `start_post_comp` means the test starts after the prior test completes
- `end_post_comp` means the test ends when the prior test completes


## Configuring HTTP Data Repository
The HTTP Data Repository is currently supported for Slurm systems only. To enable access, you must update your system schema file and create a credential file in your CloudAI project's root directory.

### Step 1: Update the System Schema File
Add the following section to your system schema TOML file (e.g., `system_schema.toml`):

```toml
[data_repository]
endpoint = "https://my-data-endpoint.com"
```

Replace the endpoint with your actual data repository URL.

### Step 2: Create the Credential File
In the root of your CloudAI project (i.e., the current working directory), create a file named `.cloudai.toml` with the following content:

```toml
[data_repository]
token = "<your-api-token-here>"
```

Replace `<your-api-token-here>` with your actual token.

### Step 3: Usage
Both the endpoint and token must be valid for the HTTP Data Repository to function correctly. If either is missing or incorrect, data will not be posted.


## Downloading and Installing the NeMo Dataset (The Pile Dataset)
This section describes how you can download the NeMo datasets on your server. The install mode of CloudAI handles the installation of all test prerequisites, but downloading and installing datasets is not the responsibility of the install mode. This is because any large datasets should be installed globally by the administrator and shared with multiple users, even if a user does not use CloudAI.

For CloudAI users, we provide a detailed guide about downloading and installing the NeMo datasets in this section. By default, the NeMo launcher uses mock datasets for testing purposes. If you want to run tests using real datasets, you must download the datasets and update the test `.toml` files accordingly to locate the datasets and provide appropriate prefixes.

 To understand the datasets available in the NeMo framework, you can refer to the Data Preparation section of [the document](https://docs.nvidia.com/launchpad/ai/base-command-nemo/latest/bc-nemo-step-02.html#use-bignlp-to-download-and-prepare-the-pile-dataset). According to the document, you can download and use the Pile dataset. The document also provides detailed instructions on how to download these datasets for various platforms.

  Let’s assume that we have a Slurm cluster.

You can download the datasets with the following command:
```bash
$ git clone https://github.com/NVIDIA/NeMo-Framework-Launcher.git
$ cd NeMo-Framework-Launcher
$ python3 launcher_scripts/main.py \
    container=nvcr.io/ea-bignlp/ga-participants/nemofw-training:23.11\
    stages=["data_preparation"]\
    launcher_scripts_path=$PWD/launcher_scripts\
    base_results_dir=$PWD/result\
    env_vars.TRANSFORMERS_OFFLINE=0\
    data_dir=directory_path_to_download_dataset\
    data_preparation.run.time_limit="96:00:00"
```

Once you submit a NeMo job with the data preparation stage, you should be able to find data downloading jobs with the squeue command. If this command does not work, please review the log files under $PWD/result. If you want to download the full Pile dataset, you should have at least 1TB of space in the directory to download the dataset because the Pile dataset size is 800GB.
By default, NeMo will look at the configuration file under conf/config.yaml:
```
defaults:
  - data_preparation: baichuan2/download_baichuan2_pile

stages:
  - data_preparation
```

As the data preparation field points to baichuan2/download_baichuan2_pile, it will read the YAML file:
```
run:
  name: download_baichuan2_pile
  results_dir: ${base_results_dir}/${.name}
  time_limit: "4:00:00"
  dependency: "singleton"
  node_array_size: 30
  array: ${..file_numbers}
  bcp_preproc_npernode: 2 # 2 should be safe to use and x2 times faster.

dataset: pile
download_the_pile: True  # Whether to download the pile dataset from the internet.
the_pile_url: "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/"  # Source URL to download The Pile dataset from.
file_numbers: "0-29"  # The pile dataset consists of 30 files (0-29), choose which ones to download.
preprocess_data: True  # True to preprocess the data from a jsonl file, False otherwise.
download_tokenizer_url: "https://huggingface.co/baichuan-inc/Baichuan2-7B-Base/blob/main/tokenizer.model"
tokenizer_typzer_library: "sentencepiece"
tokenizer_save_dir: ${data_dir}/baichuan2
tokenizer_model:  ${.tokenizer_save_dir}/baichuan2_tokenizer.model
rm_downloaded: False # Extract script will remove downloaded zst after extraction
rm_extracted: False # Preprocess script will remove extracted files after preproc.
```

You can update the fields to adjust the behavior. For example, you can update the file_numbers field to adjust the number of dataset files to download. This will allow you to save disk space.

### Note: For running Nemo Llama model, it is important to follow these additional steps:
1. Go to https://huggingface.co/docs/transformers/en/model_doc/llama#usage-tips.
2. Follow the instructions under 'Usage Tips' on how to download the tokenizer.
3. Replace "training.model.tokenizer.model=TOKENIZER_MODEL" with "training.model.tokenizer.model=YOUR_TOKENIZER_PATH" (the tokenizer should be a .model file) in conf/common/test/llama.toml.


## Using Test Hooks in CloudAI

A test hook in CloudAI is a specialized test that runs either before or after each main test in a scenario, providing flexibility to prepare the environment or clean up resources. Hooks are defined as pre-test or post-test and referenced in the test scenario’s TOML file using `pre_test` and `post_test` fields.
```
name = "nccl-test"

pre_test = "nccl_test_pre"
post_test = "nccl_test_post"

[[Tests]]
id = "Tests.1"
test_name = "nccl_test_all_reduce"
num_nodes = "2"
time_limit = "00:20:00"
```

CloudAI organizes hooks in a dedicated directory structure:

- Hook directory: All hook configurations reside in `conf/hook/`
- Hook test scenarios: Place pre-test hook and post-test hook scenario files in `conf/hook/`
- Hook tests: Place individual tests referenced in hooks within `conf/hook/test/`

In the execution flow, pre-test hooks run before the main test, which only executes if the pre-test completes successfully. Post-test hooks follow the main test, provided the prior steps succeed. If a pre-test hook fails, the main test and its post-test hook are skipped.
```
name = "nccl_test_pre"

[[Tests]]
id = "Tests.1"
test_name = "nccl_test_all_reduce"
time_limit = "00:20:00"
```

## Downloading DeepSeek Weights
To run DeepSeek R1 tests in CloudAI, you must download the model weights in advance. These weights are distributed via the NVIDIA NGC Registry and must be manually downloaded using the NGC CLI.

### Step 1: Install NGC CLI
Download and install the NGC CLI using the following commands:

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.64.2/files/ngccli_linux.zip -O ngccli_linux.zip
unzip ngccli_linux.zip
chmod u+x ngc-cli/ngc
echo "export PATH=\"$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile
```

This will make the `ngc` command available in your terminal.

### Step 2: Configure NGC CLI
Authenticate your CLI with your NGC API key by running:

```bash
ngc config set
```

When prompted, paste your API key, which you can obtain from [https://org.ngc.nvidia.com/setup](https://org.ngc.nvidia.com/setup).

### Step 3: Download the Weights
Navigate to the directory where you want the DeepSeek model weights to be stored, then run:

```bash
ngc registry model download-version nim/deepseek-ai/deepseek-r1-instruct:hf-5dde110-nim-fp8 --dest .
```

This command will create a folder named:

```
deepseek-r1-instruct_vhf-5dde110-nim-fp8/
```

inside your current directory.

### Step 4: Verify the Download
Ensure the full model has been downloaded by checking the folder size:

```bash
du -sh deepseek-r1-instruct_vhf-5dde110-nim-fp8
```

The expected size is approximately 642 GB. If it’s significantly smaller, remove the folder and re-run the download.


## Slurm specifics

### Extra srun and sbatch arguments
CloudAI forms sbatch script and srun commands following internal rules. Users can affect the generation but setting special arguments in System TOML file.

For example (in a System TOML file):,
```toml
extra_sbatch_args = [
  "--section=4",
  "--other-arg val"
]
```
will result in sbatch file content like this:
```bash
... # CloudAI set sbatch arguments
#SBATCH --section=4
#SBATCH --other-arg val
...
```

Another example (in a System TOML file):
```toml
extra_srun_args = "--arg=val --other-arg=other-val"
```
will result in srun command inside sbatch script like this:
```bash
srun ... --arg=val --other-arg=other-val ...
```

### Container mounts
CloudAI runs all slurm jobs using containers. To simplify file system related tasks, CloudAI mounts the following directories into the container:
1. Test output directory (`<output_path>/<scenario_name_with_timestamp>/<test_name>/<iteration>`, like `results/scenario_2024-06-18_17-40-13/Tests.1/0`) is mounted as `/cloudai_run_results`.
1. Test specific mounts can be specified in TOML files:
    ```toml
    extra_container_mounts = [
      "/path/to/mount1:/path/in/container1",
      "/path/to/mount2:/path/in/container2"
    ]

    [cmd_args]
    ...
    ```
    These mounts are not verified for validity and do not override default mounts.
1. Test specific mounts can be mounted in-code.

### Head node without shared storage available on compute nodes
When compute nodes don't share file system with head node, `--enable-cache-without-check` for `run` and `dry-run` skips real check for cache existence, but still builds all paths correctly. The flow is like this:

1. _[on the head node]_ run `cloudai install`
1. _[on the head node]_ copy cache to compute nodes
1. Modify system.toml to set compute nodes' installation root
1. Run `cloudai run --enable-cache-without-check ...`

#### Dev details
`SlurmCommandGenStrategy` defines abstract method `_container_mounts(tr: TestRun)` that must be implemented by every subclass. This method is used in `SlurmCommandGenStrategy.container_mounts(tr: TestRun)` (defined as `@final`) where mounts like `/cloudai_run_results` (default mount), `TestDefinition.extra_container_mounts` (from Test TOML) and test specific mounts (defined in-code) are added.

### Nsys tracing
Users can enable Nsys tracing for any workload when running via Slurm. Note, that `nsys` should be available on the compute nodes, CloudAI doesn't manage it.

Configuration fields are:
```py
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
```
Fields with `None` value are not passed to `nsys` command.

## Troubleshooting
In this section, we will guide you through identifying the root cause of issues, determining whether they stem from system infrastructure or a bug in CloudAI. Users should closely follow the USER_GUIDE.md and README.md for installation, tests, and test scenarios.

### Identifying the Root Cause
If you encounter issues running a command, start by reading the error message to understand the root cause. We strive to make our error messages and exception messages as readable and interpretable as possible.

### System Infrastructure vs. CloudAI Bugs
To determine whether an issue is due to system infrastructure or a CloudAI bug, follow these steps:

1. **Check stdout Messages:**
   If CloudAI fails to run a test successfully, it will be indicated in the stdout messages that a test has failed.

2. **Review Log Files:**
   - Navigate to the output directory and review `debug.log`, stdout, and stderr files
   - `debug.log` contains detailed steps executed by CloudAI, including generated commands, executed commands, and error messages

3. **Analyze Error Messages:**
   By examining the error messages in the log files, you can understand the type of errors CloudAI encountered.

4. **Examine Output Directory:**
   If a test fails without explicit error messages, review the output directory of the failed test. Look for `stdout.txt`, `stderr.txt`, or any generated files to understand the failure reason.

5. **Manual Rerun of Tests:**
   - To manually rerun the test, consult the `debug.log` for the command CloudAI executed
   - Look for an `sbatch` command with a generated `sbatch` script
   - Execute the command manually to debug further

If the problem persists, please report the issue at [https://github.com/NVIDIA/cloudai/issues/new/choose](https://github.com/NVIDIA/cloudai/issues/new/choose). When you report an issue, ensure it is reproducible. Follow the issue template and provide any necessary details, such as the hash commit used, system settings, any changes in the schema files, and the command.

### Test Specific Troubleshooting Guides
In addition to the general troubleshooting steps, this section provides specific troubleshooting guides for each test used in CloudAI. These guides help you identify and resolve issues unique to each template.

#### NeMo Launcher
* If your run is not successful, please review the stderr and stdout files generated under the results directory. Within the output directory, locate the run directory, and under the run directory, you will find stderr files like log-nemo-megatron-run_[job_id].err. Please review these files for any meaningful error messages
* Trying the CloudAI-generated NeMo launcher command can be helpful as well. You can find the executed command in your stdout and in your log file (debug.log) in your current working directory. Review and run the command, and you can modify the arguments to troubleshoot the issue

#### JaxToolbox (Grok)
##### Troubleshooting Steps
If an error occurs, follow these steps sequentially:

1. **Read the Error Messages**:
    Begin by reading the error messages printed by CloudAI. We strive to make our error messages clear and informative, so they are a good starting point for troubleshooting

2. **Review `profile_stderr.txt`**: JaxToolbox operates in two stages: the profiling phase and the actual run phase. We follow the PGLE workflow as described in the [PGLE workflow documentation](https://github.com/google/paxml?tab=readme-ov-file#run-pgle-workflow-on-gpu). All stderr and stdout messages from the profiling phase are stored in `profile_stderr.txt`. If the profiling stage fails, you should find relevant error messages in this file. Attempt to understand the cause of the error from these messages.

3. **Check the Actual Run Phase**:
   If the profiling stage completes successfully, CloudAI moves on to the actual run phase. The actual run generates stdout and stderr messages in separate files for each rank. Review these files to diagnose any issues during this phase.

##### Common Errors
 **DEADLINE_EXCEEDED**:
   - When running JaxToolbox on multiple nodes, the nodes must be able to communicate to execute a training job collaboratively. The DEADLINE_EXCEEDED error indicates a failure in the connection during the initialization stage. Potential causes include:
     - Hostname resolution failure by the slave nodes
     - The port opened by the master node is not accessible by other nodes
     - Network interface malfunctions
     - Significant time gap in the initialization phase among nodes. If one node starts early while others are still loading the Docker image, this error can occur. This can happen when a Docker image is not locally cached, and all nodes try to download it from a remote registry without sufficient network bandwidth. The resulting difference in initialization times can lead to a timeout on some nodes


## Running AI Dynamo with CloudAI
CloudAI supports end-to-end inference benchmarking of large language models using [AI Dynamo](https://github.com/ai-dynamo/dynamo). This section explains how to run AI Dynamo jobs via CloudAI, beginning with setting up the environment and downloading Hugging Face model weights, and continuing through job submission and monitoring.

In particular, this section will cover:

- How to download model weights using `huggingface-cli` and configure `HUGGING_FACE_HOME`
- How to write and adjust a CloudAI test schema for AI Dynamo
- How to switch the model or scale node resources
- How to monitor the job and interpret the results

CloudAI abstracts away most of the complexity in coordinating frontend, prefill, and decode nodes for AI Dynamo. Users are responsible for downloading the model weights, configuring the appropriate environment variable, and preparing the test schema.

---

### Step 1: Download Model Weights Using Hugging Face CLI

Install the Hugging Face CLI:

```bash
$ pip install -U "huggingface_hub[cli]"
```

Log in using your Hugging Face token:

```bash
$ huggingface-cli login
```

Download the model weights and tokenizer to a HF_HOME that will serve as the Hugging Face cache:

```bash
$ export HF_HOME=/path/to/hf_home/
$ huggingface-cli download nvidia/Llama-3.1-405B-Instruct-FP8
$ huggingface-cli download hf-internal-testing/llama-tokenizer
```

You can verify the model cache using:

```bash
$ huggingface-cli scan-cache -vvv

REPO ID                             REPO TYPE REVISION                                 SIZE ON DISK NB FILES LAST_MODIFIED REFS LOCAL PATH
----------------------------------- --------- ---------------------------------------- ------------ -------- ------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------
hf-internal-testing/llama-tokenizer model     d02ad6cb9dd2c2296a6332199fa2fdca5938fef0         2.3M        5 3 days ago    main /path/to/hf_home/hub/models--hf-internal-testing--llama-tokenizer/snapshots/d02ad6cb9dd2c2296a6332199fa2fdca5938fef0
nvidia/Llama-3.1-405B-Instruct-FP8  model     a0a0bc4e698fbbe4eb184bbd62067ff195a65a39       410.1G       96 4 days ago    main /path/to/hf_home/hub/models--nvidia--Llama-3.1-405B-Instruct-FP8/snapshots/a0a0bc4e698fbbe4eb184bbd62067ff195a65a39

Done in 0.3s. Scanned 2 repo(s) for a total of 410.1G.
```

The path to the downloaded weights should be consistent with the structure expected by the Hugging Face ecosystem.

---

### Step 2: Configure `HF_HOME` in the Test Schema

Set the `HF_HOME` environment variable in the test schema file (e.g., `test.toml`) so that CloudAI can locate the model weights:

```toml
name = "llama3.1_405b_fp8"
description = "llama3.1_405b_fp8"
test_template_name = "AIDynamo"

[cmd_args]
docker_image_url = "/path/to/docker_images/dynamo_4-23.sqsh"

  [cmd_args.dynamo]
  num_prefill_nodes = 1
  num_decode_nodes = 0

  [cmd_args.genai_perf]
  served_model_name = "nvidia/Llama-3.1-405B-Instruct-FP8"
  endpoint = "v1/chat/completions"
  synthetic_input_tokens_mean = 3000
  synthetic_input_tokens_stddev = 0
  warmup_request_count = 10

[extra_env_vars]
HF_HOME = "/your/path/to/hf_home"
```

This environment variable should point to the root directory used with `--local-dir` in the download step. CloudAI will use this directory to locate and load the appropriate model weights.

---

### Step 3: Node Configuration for AI Dynamo

AI Dynamo jobs use three distinct types of nodes:

- **Frontend node**: Hosts the coordination services (`etcd`, `nats`) as well as the **frontend server** and the **request generator** (`genai-perf`)
- **Prefill node(s)**: Handle the prefill stage of inference
- **Decode node(s)**: Handle the decode stage of inference (optional, depending on model and setup)

The total number of nodes required must be:

```
1 (frontend) + num_prefill_nodes + num_decode_nodes
```

This total must match the `nodes` field in your test scenario schema. If the schema specifies a different number, CloudAI will not launch the job correctly. Be sure to update the test scenario accordingly when adjusting `num_prefill_nodes` or `num_decode_nodes`.

All node role assignments and orchestration are automatically managed by CloudAI.

---

### Step 4: Launching and Monitoring the Job

To run the job:

```bash
$ python cloudaix.py run --system-config conf/staging/ai_dynamo/system/oci.toml --tests-dir conf/staging/ai_dynamo/test --test-scenario conf/staging/ai_dynamo/test_scenario/ai_dynamo.toml
```

#### Option 1: Monitor via Slurm

```bash
$ watch squeue -u $(whoami)
```

#### Option 2: Monitor Output Logs

Navigate to the results directory created by CloudAI and observe the logs:

```bash
$ cd ./results/../
$ watch tail -n 4 *.txt
```

The frontend node will initially wait to allow weight loading on all nodes. Once ready, it will launch `genai-perf`, which begins generating requests to the frontend server. All servers cooperate to complete inference, and the output will appear in `stdout.txt`.

### Step 5: Review Results
After job completion, CloudAI will place the output logs and result files in the designated results directory. To analyze performance metrics and validate inference outcomes:

- Navigate to the results directory (e.g., ./results/...)
- Most importantly, open the profile_genai_perf.csv file to examine the final benchmarking results

This CSV file includes detailed metrics collected by genai-perf, such as request latency, throughput, and system utilization statistics. Use this data to evaluate the model's performance and identify potential bottlenecks or optimization opportunities.

```
Metric,avg,min,max,p99,p95,p90,p75,p50,p25
Time To First Token (ms),"1,146.31",249.48,"3,485.23","3,457.97","3,349.56","3,215.06","1,330.93",640.07,286.52
Time To Second Token (ms),26.05,0.00,133.51,96.12,36.56,34.88,34.35,33.55,1.78
Request Latency (ms),"6,406.20","5,371.47","9,608.72","9,436.13","9,046.58","9,028.16","6,549.60","5,690.23","5,493.63"
Inter Token Latency (ms),30.35,27.59,35.60,35.23,33.88,32.53,31.05,30.13,29.04
Output Sequence Length (tokens),174.45,164.00,187.00,186.22,183.10,180.10,177.00,174.00,171.75
Input Sequence Length (tokens),"3,000.05","2,999.00","3,001.00","3,001.00","3,001.00","3,000.00","3,000.00","3,000.00","3,000.00"

Metric,Value
Output Token Throughput (per sec),261.25
Request Throughput (per sec),1.50
Request Count (count),40.00
```
