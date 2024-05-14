## Introduction
CloudAI allows users to package workloads as test templates to facilitate the automation of running experiments. This method involves packaging workloads as docker images, which is one of several approaches you can take with CloudAI. This guide details how to package a workload as a docker image and then create a test template in CloudAI.

### Step 1: Creating a Docker Image
1. **Set Up the GitLab Repository**
   Start by setting up a repository on GitLab to host your docker image. For this example, use `gitlab.com/cloudai/nccl-test`.

2. **Writing the Dockerfile**
   The dockerfile needs to specify the base image and detail the steps to add and compile the new NCCL test:
   ```dockerfile
   FROM nvcr.io/nvidia/pytorch:24.02-py3
   COPY nccl_bisection.c .
   RUN mpicc -o /usr/local/bin/bisection_mpi -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 nccl_bisection.c -lnccl -lcudart
   RUN rm nccl_bisection.c
   ```

3. **Build and Push the Docker Image**
   Build the docker image with the dockerfile and upload it to the designated repository:
   ```bash
   docker build -t gitlab.com/cloudai/nccl-test .
   docker push gitlab.com/cloudai/nccl-test
   ```

### Step 2: Verify the Docker Image
Test the docker image by running it with `srun` to verify the newly added test executes correctly:
   ```bash
   srun \
      --mpi=pmix \
      --container-image=gitlab.com/cloudai/nccl-test \
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

## Setting Up the Test Template in CloudAI
### Directory Structure and Expected Files
Create a new directory under `cloudai/schema/test_template` for organizing the new test template. This structure typically includes various Python modules to define the operational parameters of the test within CloudAI:
   ```plaintext
   $ tree cloudai/schema/test_template
   cloudai/schema/test_template
   ├── nccl_test
   │   ├── __init__.py
   │   ├── grading_strategy.py
   │   ├── command_gen_strategy.py
   │   ├── install_strategy.py
   │   └── template.py
   ```

### Implementation of Strategies
1. **Template Class**

   Define the `NcclTest` class to enable CloudAI to recognize and instantiate this as a test template. You can declare variables used within the template.
   ```python
   class NcclTest(TestTemplate):
       SUPPORTED_SUBTESTS = ["bisection_perf_mpi", ...]
   ```

2. **Install Strategy**

    Implement an installation strategy to handle the setup of necessary components on the system where the test will run. Decorators in Python are used here to register the strategy with a strategy registry. This approach allows CloudAI to dynamically recognize and use the appropriate strategy based on the test template and system configuration.
   ```python
   @StrategyRegistry.strategy(InstallStrategy, [SlurmSystem], [NcclTest])
   class NcclTestSlurmInstallStrategy(SlurmInstallStrategy):
       def install(self):
           # Logic to install the test template
       def uninstall(self):
           # Logic to uninstall the test template
   ```

3. **Command Generation Strategy**

   Develop a command generation strategy that formulates the commands needed to execute:
   ```python
   @StrategyRegistry.strategy(CommandGenStrategy, [SlurmSystem], [NcclTest])
   class NcclTestSlurmCommandGenStrategy(SlurmCommandGenStrategy):
       def gen_exec_command(self, env_vars, cmd_args):
           # Command to execute the test template
   ```

4. **cloudai/schema/test_template/\_\_init\_\_.py**

    Update \_\_init\_\_.py accordingly.
    ```python
    ....
    from .nccl_test import (
        NcclTest,
        NcclTestGradingStrategy,
        NcclTestReportGenerationStrategy,
        NcclTestSlurmCommandGenStrategy,
        NcclTestSlurmInstallStrategy,
    )
    ...

    __all__ = [
        ...
        "NcclTest",
        "NcclTestGradingStrategy",
        "NcclTestReportGenerationStrategy",
        "NcclTestSlurmCommandGenStrategy",
        "NcclTestSlurmInstallStrategy",
        ...
    ]
    ```

### Testing Procedures
After implementing a test template, you can proceed with CloudAI commands to install, dry-run, run, generate reports, and uninstall the setup to ensure proper operation.
   ```bash
   python main.py\
      --mode install\
      --system_config_path conf/system/example_slurm_cluster.toml

   python main.py\
      --mode dry-run\
      --system_config_path conf/system/example_slurm_cluster.toml\
      --test_scenario_path conf/test_scenario/nccl_test/test_scenario.toml

   python main.py\
      --mode run\
      --system_config_path conf/system/example_slurm_cluster.toml\
      --test_scenario_path conf/test_scenario/nccl_test/test_scenario.toml

   python main.py\
      --mode generate-report\
      --system_config_path conf/system/example_slurm_cluster.toml\
      --output_path /path/to/output_directory

   python main.py\
      --mode uninstall\
      --system_config_path conf/system/example_slurm_cluster.toml
   ```
