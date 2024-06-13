## Introduction
CloudAI allows users to package workloads as test templates to facilitate the automation of running experiments. This method involves packaging workloads as docker images, which is one of several approaches you can take with CloudAI. Users can run workloads using test templates. However, since docker images are not part of the CloudAI distribution, users must build their own docker image. This guide describes how to build a docker image and then run experiments.

### Step 1: Create a Docker Image
1. **Set Up the GitLab Repository**
   Start by setting up a repository on GitLab to host your docker image. For this example, use `gitlab-url.com/cloudai/nccl-test`.

2. **Writing the Dockerfile**
   The Dockerfile needs to specify the base image and detail the steps:
   ```
   FROM nvcr.io/nvidia/pytorch:24.02-py3
   ```

3. **Build and Push the Docker Image**
   Build the docker image with the Dockerfile and upload it to the designated repository:
   ```
   docker build -t gitlab-url.com/cloudai/nccl-test .
   docker push gitlab-url.com/cloudai/nccl-test
   ```

### Step 2: Verify the Docker Image
Test the docker image by running it with `srun` to verify that the docker image runs correctly:
   ```
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

### Step 3: Update Test Template File
You can find an NCCL test template file under `conf/test_template/nccl_test.toml`, where you will find all arguments for the NCCL test as shown below.
```
name = "NcclTest"

[cmd_args]
  [cmd_args.docker_image_url]
  type = "str"
  default = "DOCKER_IMAGE_URL"

  [cmd_args.subtest_name]
  type = "preset"
  values = ["all_reduce_perf_mpi",
            "all_gather_perf_mpi",
            "alltoall_perf_mpi",
            "broadcast_perf_mpi",
            "gather_perf_mpi",
            "hypercube_perf_mpi",
            "reduce_perf_mpi",
            "reduce_scatter_perf_mpi",
            "scatter_perf_mpi",
            "sendrecv_perf_mpi",
            "bisection_perf_mpi"]
  default = "all_reduce_perf_mpi"
```
Please update the docker image URL by updating the default docker image URL path from DOCKER_IMAGE_URL to gitlab-url.com/cloudai/nccl-test.

### Step 4: Install CloudAI
You can repeat the same for other docker images as well. Afterwards, please install CloudAI with the following command.
```
python main.py\
    --mode install\
    --system_config_path conf/system/example_slurm_cluster.toml
```

### Step 5: Run Experiments and Generate Reports
You can find predefined NCCL test schemas under `conf/test` and a test scenario at `conf/nccl_test.toml`. In a test schema file, you can adjust arguments as shown below. In the cmd_args section, you can provide different values other than the default values for each argument. In extra_cmd_args, you can provide additional arguments that will be appended after the NCCL test command. You can specify additional environment variables in the extra_env_vars section.
```
name = "nccl_test_bisection"
description = "Bisection"
test_template_name = "NcclTest"
extra_cmd_args = "--stepfactor 2"

[cmd_args]
"subtest_name" = "bisection_perf_mpi"
"ngpus" = "1"
"minbytes" = "128"
"maxbytes" = "4G"
"iters" = "100"
"warmup_iters" = "50"
```

You can find part of a test scenario below. You should define the name of the test scenario, and describe tests and their dependencies. The name of the tests should be found in the test schema files. Node lists and time limits are optional. Nodes should be described as a list of node names as shown in a Slurm system. Alternatively, if groups are defined in the system schema, you can ask CloudAI to allocate a specific number of nodes from a specified partition and group. In the example below, 16 nodes are allocated from a group, GROUP, from a partition, PARTITION. There are three types of dependencies: start_post_comp, start_post_init, end_post_comp. All dependencies are described as a pair of the depending test name and delay. The name should be taken from the test name as shown in the test scenario. The delay is described in the number of seconds. start_post_comp means that the current test should be started after a specific delay of the completion of the depending test. start_post_init means that the current test should start after the start of the depending test. end_post_comp means that the current test should be completed after the completion of the depending test.
```toml
name = "nccl-test"

[Tests.1]
  name = "nccl_test_all_reduce"
  nodes = ['PARTITION:GROUP:16']
  time_limit = "00:20:00"

[Tests.2]
  name = "nccl_test_all_gather"
  nodes = ['PARTITION:GROUP:16']
  time_limit = "00:20:00"
  [Tests.2.dependencies]
    start_post_comp = { name = "Tests.1", time = 0 }
```

To generate NCCL test commands without actual execution, use the dry-run mode. You can review debug.log created under the current working directory to see the generated commands from CloudAI. Please note that group node allocations are not currently supported in the dry-run mode.
```bash
python main.py\
    --mode dry-run\
    --system_config_path conf/system/example_slurm_cluster.toml\
    --test_scenario_path conf/test_scenario/nccl_test.toml
```

You can run NCCL test experiments with the following command. Whenever you run CloudAI in the run mode, a new directory will be created under the results directory with the timestamp. In the directory, you can find the results from the test scenario including stdout and stderr. Once completed successfully, you can find generated reports under the directories as well.
```bash
python main.py\
    --mode run\
    --system_config_path conf/system/example_slurm_cluster.toml\
    --test_scenario_path conf/test_scenario/nccl_test.toml
```
