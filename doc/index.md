# CloudAI Benchmark Framework

CloudAI benchmark framework aims to develop an industry standard benchmark focused on grading Data Center (DC) scale AI systems in the Cloud. The primary motivation is to provide automated benchmarking on various systems.

## Get Started
```bash
git clone git@github.com:NVIDIA/cloudai.git
cd cloudai
uv run cloudai --help
```

**Note**: instructions for setting up access for `enroot` are available [installation guide](./workloads_requirements_installation.rst).

### `pip`-based installation
See required Python version in the `.python-version` file, please ensure you have it installed (see how a custom python version [can be installed](#install-custom-python-version)). Follow these steps:
```bash
git clone git@github.com:NVIDIA/cloudai.git
cd cloudai
python -m venv venv
source venv/bin/activate
pip install -e .
```

(install-custom-python-version)=
### Install custom python version
If your system python version is not supported, you can install a custom version using [uv](https://docs.astral.sh/uv/getting-started/installation/) tool:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --seed  # this will pick up the python version from .python-version file
                # --seed will install pip and setuptools
source .venv/bin/activate
```


## Key Concepts
CloudAI operates on four main schemas:

- **System Schema**: Describes the system, including the scheduler type, node list, and global environment variables.
- **Test Schema**: An instance of a test template with custom arguments and environment variables.
- **Test Scenario Schema**: A set of tests with dependencies and additional descriptions about the test scenario.

These schemas enable CloudAI to be flexible and compatible with different systems and configurations.


## Support matrix
|Test|Slurm|Kubernetes|RunAI|Standalone|
|---|---|---|---|---|
|ChakraReplay|✅|❌|❌|❌|
|GPT|✅|❌|❌|❌|
|Grok|✅|❌|❌|❌|
|NCCL|✅|✅|✅|❌|
|NeMo Launcher|✅|❌|❌|❌|
|NeMo Run|✅|❌|❌|❌|
|Nemotron|✅|❌|❌|❌|
|Sleep|✅|✅|❌|✅|
|UCC|✅|❌|❌|❌|
|SlurmContainer|✅|❌|❌|❌|
|MegatronRun (experimental)|✅|❌|❌|❌|



## CloudAI Modes Usage Examples

CloudAI supports five modes:
- [install](install) - Use the install mode to install all test templates in the specified installation path 
- [dry-run](dry-run) - Use the dry-run mode to simulate running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups
- [run](run) - Use the run mode to run experiments
- [generate-report](generate-report) - Use the generate-report mode to generate reports under the test directories alongside the raw data
- [uninstall](uninstall) - Use the uninstall mode to remove installed test templates

(install)=
### install

To install test prerequisites, run CloudAI CLI in install mode. For more details, please refer to the [installation guide](./workloads_requirements_installation.rst).

Please make sure to use the correct system configuration file that corresponds to your current setup for installation and experiments.
```bash
cloudai install\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(dry-run)=
### dry-run
To simulate running experiments without execution, use the dry-run mode:
```bash
cloudai dry-run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(run)=
### run
To run experiments, execute CloudAI CLI in run mode:
```bash
cloudai run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(generate-report)=
### generate-report
To generate reports, execute CloudAI CLI in generate-report mode:
```bash
cloudai generate-report\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml\
    --result-dir /path/to/result_directory
```
In the generate-report mode, use the `--result-dir` argument to specify a subdirectory under the output directory.
This subdirectory is usually named with a timestamp for unique identification.

(uninstall)=
### uninstall
To uninstall test prerequisites, run CloudAI CLI in uninstall mode:
```bash
cloudai uninstall\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(verify-configs)=
### verify-configs
```bash
# verify all at once
cloudai verify-configs conf

# verify a single file
cloudai verify-configs conf/common/system/example_slurm_cluster.toml

#  verify all scenarios using specific folder with Test TOMLs
cloudai verify-configs --tests-dir conf/release/spcx/l40s/test conf/release/spcx/l40s/test_scenario
```


```{toctree}
:maxdepth: 1
:caption: Contents:

workloads/index
DEV
reporting
USER_GUIDE
workloads_requirements_installation
```
