# CloudAI User Guide
This is a CloudAI user guide to help users use CloudAI, covering topics such as adding a new test template and downloading datasets for running NeMo-launcher.

## Adding a New Test Template
CloudAI allows users to package workloads as test templates to facilitate the automation of running experiments. This method involves packaging workloads as docker images, which is one of several approaches you can take with CloudAI. Users can run workloads using test templates. However, since docker images are not part of the CloudAI distribution, users must build their own docker image. This guide describes how to build a docker image and then run experiments.

#### Step 1: Create a Docker Image
1. **Set Up the GitLab Repository**
   Start by setting up a repository on GitLab to host your docker image. For this example, use `gitlab-url.com/cloudai/nccl-test`.

2. **Writing the Dockerfile**
   The Dockerfile needs to specify the base image and detail the steps:
   ```dockerfile
   FROM nvcr.io/nvidia/pytorch:24.02-py3
   ```

3. **Build and Push the Docker Image**
   Build the docker image with the Dockerfile and upload it to the designated repository:
   ```bash
   docker build -t gitlab-url.com/cloudai/nccl-test .
   docker push gitlab-url.com/cloudai/nccl-test
   ```

4. **Verify the Docker Image**
   Test the docker image by running it with `srun` to verify that the docker image runs correctly:
   ```bash
   srun \
      --mpi=pmix \
      --container-image=gitlab-url.com/cloudai/nccl-test \
      /usr/local/bin/all_reduce_perf_mpi \
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

#### Step 2: Prepare configuration files
CloudAI is fully configurable via set of TOML configuration files. You can find examples of these files under `conf/`. In this guide, we will use the following configuration files:
1. `myconfig/test_templates/nccl_template.toml` - Describes the test template configuration.
1. `myconfig/system.toml` - Describes the system configuration.
1. `myconfig/tests/nccl_test.toml` - Describes the test to run.
1. `myconfig/scenario.toml` - Describes the test scenario configuration.


#### Step 3: Test Template
Test template config describes all arguments of a test. Let's create a test template file for the NCCL test. You can find more examples of test templates under `conf/test_template/`. Our example will be small for demonstration purposes. Below is the `myconfig/test_templates/nccl_template.toml` file:
```toml
name = "NcclTest"

[cmd_args]
  [cmd_args.docker_image_url]
  type = "str"
  default = "nvcr.io/nvidia/pytorch:24.02-py3"

  [cmd_args.subtest_name]
  type = "preset"
  values = ["all_reduce_perf_mpi"]
  default = "all_reduce_perf_mpi"

  [cmd_args.ngpus]
  type = "int"
  default = "1"

  [cmd_args.minbytes]
  type = "str"
  default = "32M"

  [cmd_args.maxbytes]
  type = "str"
  default = "32M"

  [cmd_args.iters]
  type = "int"
  default = "20"

  [cmd_args.warmup_iters]
  type = "int"
  default = "5"
```
Notice that `cmd_args.docker_image_url` uses `nvcr.io/nvidia/pytorch:24.02-py3`, but you can use Docker image from Step 1.

#### Step 3: System Config
System config describes the system configuration. You can find more examples of system configs under `conf/system/`. Our example will be small for demonstration purposes. Below is the `myconfig/system.toml` file:
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
  nodes = ["<nodes-[01-10]>"]
```
Please replace `<YOUR PARTITION NAME>` with the name of the partition you want to use. You can find the partition name by running `sinfo` on the cluster. Replace `<nodes-[01-10]>` with the node names you want to use.

#### Step 4: Install test requirements
Once all configs are ready, it is time to install test requirements. It is done once so that you can run multiple experiments without reinstalling the requirements. This step requires the system config file from the previous step.
```bash
cloudai --mode install \
   --system-config myconfig/system.toml \
   --test-templates-dir myconfig/test_templates/ \
   --tests-dir myconfig/tests/
```

#### Step 5: Test Configuration
Test Config describes a particular test configuration to be run. It is based on Test Template and will be used in Test Sceanrio. Below is the `myconfig/tests/nccl_test.toml` file:
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
You can find more examples under `conf/test`. In a test schema file, you can adjust arguments as shown above. In the `cmd_args` section, you can provide different values other than the default values for each argument. In `extra_cmd_args`, you can provide additional arguments that will be appended after the NCCL test command. You can specify additional environment variables in the `extra_env_vars` section.

#### Step 6: Run Experiments
Test Scenario uses Test description from the previous step. Below is the `myconfig/scenario.toml` file:
```toml
name = "nccl-test"

[Tests.1]
  name = "nccl_test_all_reduce_single_node"
  time_limit = "00:20:00"

[Tests.2]
  name = "nccl_test_all_reduce_single_node"
  time_limit = "00:20:00"
  [Tests.2.dependencies]
    start_post_comp = { name = "Tests.1", time = 0 }
```

Notes on the test scenario:
1. `name` is a mandatory filed. Other fields describe arbitrary number of tests and their dependencies.
1. The `name` of the tests should be found in the test schema files. Node lists and time limits are optional.
1. If needed, `nodes` should be described as a list of node names as shown in a Slurm system. Alternatively, if groups are defined in the system schema, you can ask CloudAI to allocate a specific number of nodes from a specified partition and group. For example `nodes = ['PARTITION:GROUP:16']`: 16 nodes are allocated from a group `GROUP`, from a partition `PARTITION`.
1. There are three types of dependencies: `start_post_comp`, `start_post_init` and `end_post_comp`.
    1. `start_post_comp` means that the current test should be started after a specific delay of the completion of the depending test.
    1. `start_post_init` means that the current test should start after the start of the depending test.
    1. `end_post_comp` means that the current test should be completed after the completion of the depending test.

   All dependencies are described as a pair of the depending test name and a delay. The name should be taken from the test name as set in the test scenario. The delay is described in the number of seconds.


To generate NCCL test commands without actual execution, use the `dry-run` mode. You can review `debug.log` (or other file specifued with `--log-file`) to see the generated commands from CloudAI. Please note that group node allocations are not currently supported in the `dry-run` mode.
```bash
cloudai --mode dry-run \
    --test-scenario myconfig/scenario.toml \
    --system-config myconfig/system.toml \
    --test-templates-dir myconfig/test_templates/ \
    --tests-dir myconfig/tests/
```

You can run NCCL test experiments with the following command. Whenever you run CloudAI in the `run` mode, a new directory will be created under the results directory with the timestamp. In the directory, you can find the results from the test scenario including stdout and stderr. Once completed successfully, you can find generated reports under the directories as well.
```bash
cloudai --mode run \
    --test-scenario myconfig/scenario.toml \
    --system-config myconfig/system.toml \
    --test-templates-dir myconfig/test_templates/ \
    --tests-dir myconfig/tests/
```

#### Step 7: Generate Reports
Once the test scenario is completed, you can generate reports using the following command:
```bash
cloudai --mode generate-report \
   --test-scenario myconfig/scenario.toml \
   --system-config myconfig/system.toml \
   --test-templates-dir myconfig/test_templates/ \
   --tests-dir myconfig/tests/ \
   --output-dir results/2024-06-18_17-40-13/
```

`--output-dir` accepts one scenario run results directory.

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
- **install_path**: Specifies the path where test templates are installed. Docker images are downloaded to this path if the user chooses to cache Docker images.
- **output_path**: Defines the default path where outputs are stored. Whenever a user runs a test scenario, a new subdirectory will be created under this path.
- **default_partition**: Specifies the default partition where jobs are scheduled.
- **partitions**: Describes the available partitions and nodes within those partitions.
  - **groups**: Within the same partition, users can define groups of nodes. This is a logical grouping that does not overlap between groups. The group concept can be used to allocate nodes from specific groups in a test scenario schema.
- **mpi**: Indicates the Process Management Interface (PMI) implementation to be used for inter-process communication.
- **gpus_per_node** and **ntasks_per_node**: These are Slurm arguments passed to the `sbatch` script and `srun`.
- **cache_docker_images_locally**: Specifies whether CloudAI should cache remote Docker images locally during installation. If set to `true`, CloudAI will cache the Docker images, enabling local access without needing to download them each time a test template is run. This approach saves network bandwidth but requires more disk capacity. If set to `false`, CloudAI will allow Slurm to download the Docker images as needed when they are not cached locally by Slurm.
- **global_env_vars**: Lists all global environment variables that will be applied globally whenever tests are run.

## Describing a Test Scenario in the Test Scenario Schema
A test scenario is a set of tests with specific dependencies between them. A test scenario is described in a TOML schema file. This is an example of a test scenario file:
```
name = "nccl-test"

[Tests.1]
  name = "nccl_test_all_reduce"
  num_nodes = "2"
  time_limit = "00:20:00"

[Tests.2]
  name = "nccl_test_all_gather"
  num_nodes = "2"
  time_limit = "00:20:00"
  [Tests.2.dependencies]
    start_post_comp = { name = "Tests.1", time = 0 }

[Tests.3]
  name = "nccl_test_reduce_scatter"
  num_nodes = "2"
  time_limit = "00:20:00"
  [Tests.3.dependencies]
    start_post_comp = { name = "Tests.2", time = 0 }
```

The `name` field is the test scenario name, which can be any unique identifier for the scenario. Each test has a section name, following the convention `Tests.1`, `Tests.2`, etc., with an increasing index. The `name` of a test should be specified in this section and must correspond to an entry in the test schema. If a test in a test scenario is not present in the test schema, CloudAI will not be able to identify it.

There are two ways to specify nodes. The first is using the `num_nodes` field as shown in the example. The second is specifying nodes explicitly like `nodes = ["node-001", "node-002"]`. Alternatively, you can utilize the groups feature in the system schema to specify nodes like `nodes = ['PARTITION_NAME:GROUP_NAME:NUM_NODES']`, which allocates `num_nodes` from the group name in the specified partition.

You can optionally specify a time limit in the Slurm format. Tests can have dependencies. If no dependencies are specified, all tests will run in parallel. CloudAI supports three types of dependencies: `start_post_init`, `start_post_comp`, and `end_post_comp`.

Dependencies of a test can be described as a subsection of the test. The dependency name should appear first, followed by a dictionary as its value. The dictionary has two fields: `name` and `time`. The `name` field specifies the test that the current test depends on, and the `time` field specifies the delay in seconds.

- `start_post_init` means the test starts after the prior test begins, with a specified delay.
- `start_post_comp` means the test starts after the prior test completes.
- `end_post_comp` means the test ends when the prior test completes.


## Downloading and Installing the NeMo Dataset (The Pile Dataset)
This section describes how you can download the NeMo datasets on your server. The install mode of CloudAI handles the installation of all test templates, but downloading and installing datasets is not the responsibility of the install mode. This is because any large datasets should be installed globally by the administrator and shared with multiple users, even if a user does not use CloudAI. For CloudAI users, we provide a detailed guide about downloading and installing the NeMo datasets in this section. To understand the datasets available in the NeMo framework, you can refer to the Data Preparation section of [the document](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/baichuan2/dataprep.html). According to the document, you can download and use the Pile dataset. The document also provides detailed instructions on how to download these datasets for various platforms. Let’s assume that we have a Slurm cluster.

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
3. Replace "training.model.tokenizer.model=TOKENIZER_MODEL" with "training.model.tokenizer.model=YOUR_TOKENIZER_PATH" (the tokenizer should be a .model file) in conf/general/test/llama.toml.

## Troubleshooting
In this section, we will guide you through identifying the root cause of issues, determining whether they stem from system infrastructure or a bug in CloudAI. Users should closely follow the USER_GUIDE.md and README.md for installation, adding test templates, tests, and test scenarios.

### Identifying the Root Cause
If you encounter issues running a command, start by reading the error message to understand the root cause. We strive to make our error messages and exception messages as readable and interpretable as possible.

### System Infrastructure vs. CloudAI Bugs
To determine whether an issue is due to system infrastructure or a CloudAI bug, follow these steps:

1. **Check stdout Messages**
   If CloudAI fails to run a test template successfully, it will be indicated in the stdout messages that a test has failed.

2. **Review Log Files**
   - Navigate to the output directory and review `debug.log`, stdout, and stderr files.
   - `debug.log` contains detailed steps executed by CloudAI, including generated commands, executed commands, and error messages.

3. **Analyze Error Messages**
   By examining the error messages in the log files, you can understand the type of errors CloudAI encountered.

4. **Examine Output Directory**
   If a test fails without explicit error messages, review the output directory of the failed test. Look for `stdout.txt`, `stderr.txt`, or any generated files to understand the failure reason.

5. **Manual Rerun of Tests**
   - To manually rerun the test, consult the `debug.log` for the command CloudAI executed.
   - Look for an `sbatch` command with a generated `sbatch` script.
   - Execute the command manually to debug further.

If the problem persists, please report the issue at [https://github.com/NVIDIA/cloudai/issues/new/choose](https://github.com/NVIDIA/cloudai/issues/new/choose). When you report an issue, ensure it is reproducible. Follow the issue template and provide any necessary details, such as the hash commit used, system settings, any changes in the schema files, and the command.

### Test Template-Specific Troubleshooting Guides
In addition to the general troubleshooting steps, this section provides specific troubleshooting guides for each test template used in CloudAI. These guides help you identify and resolve issues unique to each template.

#### NeMo Launcher
* If your run is not successful, please review the stderr and stdout files generated under the results directory. Within the output directory, locate the run directory, and under the run directory, you will find stderr files like log-nemo-megatron-run_[job_id].err. Please review these files for any meaningful error messages.
* Trying the CloudAI-generated NeMo launcher command can be helpful as well. You can find the executed command in your stdout and in your log file (debug.log) in your current working directory. Review and run the command, and you can modify the arguments to troubleshoot the issue.

#### JaxToolbox (Grok)
##### Troubleshooting Steps
If an error occurs, follow these steps sequentially:

1. **Read the Error Messages**:
   - Begin by reading the error messages printed by CloudAI. We strive to make our error messages clear and informative, so they are a good starting point for troubleshooting.

2. **Review `profile_stderr.txt`**:
   - JaxToolbox operates in two stages: the profiling phase and the actual run phase. We follow the PGLE workflow as described in the [PGLE workflow documentation](https://github.com/google/paxml?tab=readme-ov-file#run-pgle-workflow-on-gpu). All stderr and stdout messages from the profiling phase are stored in `profile_stderr.txt`. If the profiling stage fails, you should find relevant error messages in this file. Attempt to understand the cause of the error from these messages.

3. **Check the Actual Run Phase**:
   - If the profiling stage completes successfully, CloudAI moves on to the actual run phase. The actual run generates stdout and stderr messages in separate files for each rank. Review these files to diagnose any issues during this phase.

##### Common Errors
1. **DEADLINE_EXCEEDED**:
   - When running JaxToolbox on multiple nodes, the nodes must be able to communicate to execute a training job collaboratively. The DEADLINE_EXCEEDED error indicates a failure in the connection during the initialization stage. Potential causes include:
     - Hostname resolution failure by the slave nodes.
     - The port opened by the master node is not accessible by other nodes.
     - Network interface malfunctions.
     - Significant time gap in the initialization phase among nodes. If one node starts early while others are still loading the Docker image, this error can occur. This can happen when a Docker image is not locally cached, and all nodes try to download it from a remote registry without sufficient network bandwidth. The resulting difference in initialization times can lead to a timeout on some nodes.
