# CloudAI Benchmark Framework

CloudAI benchmark framework aims to develop an industry standard benchmark focused on grading Data Center (DC) scale AI systems in the Cloud. The primary motivation is to provide automated benchmarking on various systems.

## Get Started

Using `uv` tool allows users to run CloudAI without manually managing required Python versions and dependencies.
```bash
git clone git@github.com:NVIDIA/cloudai.git
cd cloudai
uv run cloudai --help
```

Please refer to the [installation guide](https://nvidia.github.io/cloudai/workloads_requirements_installation.html) for details on setting up workloads' requirements.

For details and `pip`-based installation, please refer to the [documentation](https://nvidia.github.io/cloudai/#get-started).

## Key Concepts

CloudAI operates on four main schemas:

- **System Schema**: Describes the system, including the scheduler type, node list, and global environment variables.
- **Test Schema**: An instance of a test template with custom arguments and environment variables.
- **Test Scenario Schema**: A set of tests with dependencies and additional descriptions about the test scenario.

These schemas enable CloudAI to be flexible and compatible with different systems and configurations.


## Support matrix
|Test|Slurm|Kubernetes|RunAI|Standalone|
|---|---|---|---|---|
|AI Dynamo|✅|✅|❌|❌|
|BashCmd|✅|❌|❌|❌|
|ChakraReplay|✅|❌|❌|❌|
|DDLB|✅|❌|❌|❌|
|DeepEP|✅|❌|❌|❌|
|Jax GPT (deprecated)|✅|❌|❌|❌|
|Jax Grok (deprecated)|✅|❌|❌|❌|
|Jax Nemotron (deprecated)|✅|❌|❌|❌|
|MegatronRun|✅|❌|❌|❌|
|NCCL|✅|✅|✅|❌|
|NeMo v1.0 (aka NemoLauncher) (deprecated)|✅|❌|❌|❌|
|NeMo v2.0 (aka NemoRun)|✅|❌|❌|❌|
|NIXL benchmark|✅|❌|❌|❌|
|NIXL kvbench|✅|❌|❌|❌|
|NIXL perftest|✅|❌|❌|❌|
|Sleep|✅|✅|❌|✅|
|SlurmContainer|✅|❌|❌|❌|
|Triton Inference|✅|❌|❌|❌|
|UCC|✅|❌|❌|❌|

*deprecated means that a workload support exists, but we are not maintaining it actively anymore and newer configurations might not work.

For more detailed information, please refer to the [official documentation](https://nvidia.github.io/cloudai/workloads/index.html).

## CloudAI Modes Usage Examples

CloudAI supports five modes:
- [install](#install) - Use the install mode to install all test templates in the specified installation path 
- [dry-run](#dry-run) - Use the dry-run mode to simulate running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups
- [run](#run) - Use the run mode to run experiments
- [generate-report](#generate-report) - Use the generate-report mode to generate reports under the test directories alongside the raw data
- [uninstall](#uninstall) - Use the uninstall mode to remove installed test templates

### install

To install test prerequisites, run CloudAI CLI in install mode. For more details, please refer to the [installation guide](https://nvidia.github.io/cloudai/workloads_requirements_installation.html).

Please make sure to use the correct system configuration file that corresponds to your current setup for installation and experiments.
```bash
cloudai install\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```
### dry-run
To simulate running experiments without execution, use the dry-run mode:
```bash
cloudai dry-run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```
### run
To run experiments, execute CloudAI CLI in run mode:
```bash
cloudai run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```
### generate-report
To generate reports, execute CloudAI CLI in generate-report mode:
```bash
cloudai generate-report\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml\
    --result-dir /path/to/result_directory
```
In the generate-report mode, use the --result-dir argument to specify a subdirectory under the output directory.
This subdirectory is usually named with a timestamp for unique identification.
### uninstall
To uninstall test prerequisites, run CloudAI CLI in uninstall mode:
```bash
cloudai uninstall\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```
### verify-configs
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
