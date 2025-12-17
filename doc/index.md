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

## CloudAI Modes Usage Examples

Global options for `cloudai` command:
- `--log-file <path>`: specify a file to log output, be default `debug.log` in the current directory is used. Contains log entries of level `DEBUG` and higher.
- `--log-level <level>`: specify logging level for standard output, default is `INFO`.

(run)=
### run
This mode runs workloads. It automatically installs prerequisites if they are not met.

```bash
cloudai run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(dry-run)=
### dry-run
This mode simulates running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups.

```bash
cloudai dry-run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(generate-report)=
### generate-report
This mode generates reports under the scenario directory. It automatically runs as part of the `run` mode after experiments are completed.

```bash
cloudai generate-report\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml\
    --result-dir /path/to/result_directory
```

(install)=
### install
This mode installs test prerequisites. For more details, please refer to the [installation guide](https://nvidia.github.io/cloudai/workloads_requirements_installation.html). It automatically runs as part of the `run` mode if prerequisites are not met.

```bash
cloudai install\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(uninstall)=
### uninstall
The opposite to the install mode, this mode removes installed test prerequisites.

```bash
cloudai uninstall\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

(list)=
### list
This mode lists internal components available within CloudAI.
```bash
cloudai list <component_type>
```

(verify-configs)=
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


```{toctree}
:maxdepth: 1
:caption: Contents:
:glob:

*
workloads/index
```
