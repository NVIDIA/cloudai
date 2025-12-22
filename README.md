# CloudAI Benchmark Framework

CloudAI benchmark framework aims to develop an industry standard benchmark focused on grading Data Center (DC) scale AI systems in the Cloud. The primary motivation is to provide automated benchmarking on various systems.

## Get Started

Using `uv` tool allows users to run CloudAI without manually managing required Python versions and dependencies.
```bash
git clone git@github.com:NVIDIA/cloudai.git
cd cloudai
uv run cloudai --help
```

For details on setting up workloads' requirements, please refer to the [installation guide](https://nvidia.github.io/cloudai/workloads_requirements_installation.html)

For details and `pip`-based installation, please refer to the [documentation](https://nvidia.github.io/cloudai/#get-started).

## Key Concepts

CloudAI operates on three main schemas:

- **System Schema**: Describes the system, including the scheduler type, node list, and global environment variables
- **Test Schema**: An instance of a test template with custom arguments and environment variables
- **Test Scenario Schema**: A set of tests with dependencies and additional descriptions about the test scenario

These schemas enable CloudAI to be flexible and compatible with different systems and configurations.


## Support Matrix
|Test|Slurm|Kubernetes|RunAI|Standalone|
|---|---|---|---|---|
|AI Dynamo|✅|✅|❌|❌|
|BashCmd|✅|❌|❌|❌|
|ChakraReplay|✅|❌|❌|❌|
|DDLB|✅|❌|❌|❌|
|DeepEP|✅|❌|❌|❌|
|JaxToolbox workloads (DEPRECATED)|✅|❌|❌|❌|
|MegatronRun|✅|❌|❌|❌|
|NCCL|✅|✅|✅|❌|
|NeMo v1.0 aka NemoLauncher (DEPRECATED)|✅|❌|❌|❌|
|NeMo v2.0 (aka NemoRun)|✅|❌|❌|❌|
|NIXL benchmark|✅|❌|❌|❌|
|NIXL kvbench|✅|❌|❌|❌|
|NIXL CTPerf|✅|❌|❌|❌|
|Sleep|✅|✅|❌|✅|
|SlurmContainer|✅|❌|❌|❌|
|Triton Inference|✅|❌|❌|❌|
|UCC|✅|❌|❌|❌|

Note: Deprecated means that a workload support exists, but we are not maintaining it actively anymore and newer configurations might not work.

For more detailed information, please refer to the [official documentation](https://nvidia.github.io/cloudai/workloads/index.html).

## CloudAI Modes Usage Examples

### run
This mode runs workloads. It automatically installs prerequisites if they are not met.

```bash
cloudai run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

### dry-run
This mode simulates running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups.

```bash
cloudai dry-run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

### generate-report
This mode generates reports under the scenario directory. It automatically runs as part of the `run` mode after experiments are completed.

```bash
cloudai generate-report\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml\
    --result-dir /path/to/result_directory
```

### install
This mode installs test prerequisites. For more details, please refer to the [installation guide](https://nvidia.github.io/cloudai/workloads_requirements_installation.html). It automatically runs as part of the `run` mode if prerequisites are not met.

```bash
cloudai install\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

### uninstall
The opposite to the install mode, this mode removes installed test prerequisites.

```bash
cloudai uninstall\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

### list
This mode lists internal components available within CloudAI.
```bash
cloudai list <component_type>
```

### verify-configs
This mode verifies the correctness of system, test and test scenario configuration files.

```bash
# verify all at once
cloudai verify-configs conf

# verify a single file
cloudai verify-configs conf/common/system/example_slurm_cluster.toml

#  verify all scenarios using specific folder with Test TOMLs
cloudai verify-configs --tests-dir conf/release/spcx/l40s/test conf/release/spcx/l40s/test_scenario
```

## Additional Documentation
For more detailed instructions and guidance, including advanced usage and troubleshooting, please refer to the [official documentation](https://nvidia.github.io/cloudai/).

## Contribution
Please feel free to contribute to the CloudAI project and share your insights. Your contributions are highly appreciated.

## License
This project is licensed under Apache 2.0. See the LICENSE file for detailed information.
