## Introduction
CloudAI allows users to package workloads as test templates to facilitate the automation of running experiments. This method involves packaging workloads as docker images, which is one of several approaches you can take with CloudAI. Users can run workloads using test templates. However, since docker images are not part of the CloudAI distribution, users must build their own docker image. This guide describes how to build a docker image and then run experiments.

### Step 1: Create a Docker Image
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

### Step 2: Prepare configuration files
CloudAI is fully configurable via set of TOML configuration files. You can find examples of these files under `conf/`. In this guide, we will use the following configuration files:
1. `myconfig/test_templates/nccl_template.toml` - Describes the test template configuration.
1. `myconfig/system.toml` - Describes the system configuration.
1. `myconfig/tests/nccl_test.toml` - Describes the test to run.
1. `myconfig/scenario.toml` - Describes the test scenario configuration.


### Step 3: Test Template
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

### Step 3: System Config
System config describes the system configuration. You can find more examples of system configs under `conf/system/`. Our example will be small for demonstration purposes. Below is the `myconfig/system.toml` file:
```toml
name = "my-cluster"
scheduler = "slurm"

install_path = "./install"
output_path = "./results"
cache_docker_images_locally = "True"
default_partition = "<YOUR PARTITION NAME>"

gpus_per_node = 8
ntasks_per_node = 8

[partitions]
  [partitions.<YOUR PARTITION NAME>]
  name = "<YOUR PARTITION NAME>"
  nodes = ["<nodes-[01-10]>"]
```
Please replace `<YOUR PARTITION NAME>` with the name of the partition you want to use. You can find the partition name by running `sinfo` on the cluster. Replace `<nodes-[01-10]>` with the node names you want to use.

### Step 4: Install test requirements
Once all configs are ready, it is time to install test requirements. It is done once so that you can run multiple experiments without reinstalling the requirements. This step requires the system config file from the previous step.
```bash
cloudai --mode install \
   --system-config myconfig/system.toml \
   --test-templates-dir myconfig/test_templates/ \
   --tests-dir myconfig/tests/
```

### Step 5: Test Configuration
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

### Step 6: Run Experiments
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

### Step 7: Generate Reports
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
