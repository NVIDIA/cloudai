# CloudAI Benchmark Framework

CloudAI benchmark framework aims to develop an industry standard benchmark focused on grading Data Center (DC) scale AI systems in the Cloud. The primary motivation is to provide automated benchmarking on various systems.

## Get Started
**Note**: instructions for installing a custom python version are available [here](install-custom-python-version).

**Note**: instructions for setting up access for `enroot` are available [here](set-up-access-to-the-private-ngc-registry).

1. Clone the CloudAI repository to your local machine:
    ```bash
    git clone git@github.com:NVIDIA/cloudai.git
    cd cloudai
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. Next, install the required packages:
    ```bash
    pip install .
    ```

    For development please use the following command:
    ```bash
    pip install -e '.[dev]'
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
|NCCL|✅|✅|✅|❌|
|NeMo Launcher|✅|❌|❌|❌|
|NeMo Run|✅|❌|❌|❌|
|Sleep|✅|✅|❌|✅|
|UCC|✅|❌|❌|❌|
|SlurmContainer|✅|❌|❌|❌|
|MegatronRun (experimental)|✅|❌|❌|❌|

## Details
(set-up-access-to-the-private-ngc-registry)=
###  Set Up Access to the Private NGC Registry
First, ensure you have access to the Docker repository. Follow the following steps:

1. **Sign In**: Go to [NVIDIA NGC](https://ngc.nvidia.com/signin) and sign in with your credentials.
2. **Generate API Key**:
    - On the top right corner, click on the dropdown menu next to your profile
    - Select "Setup"
    - In the "Setup" section, find "Keys/Secrets"
    - Click "Generate API Key" and confirm when prompted. A new API key will be presented
    - **Note**: Save this API key locally as you will not be able to view it again on NGC

Next, set up your enroot credentials. Ensure you have the correct credentials under `~/.config/enroot/.credentials`:
```
machine nvcr.io login $oauthtoken password <api-key>
```
Replace `<api-key>` with your respective credentials. Keep `$oauthtoken` as is.


(install-custom-python-version)=
### Install custom python version
If your system python version is not supported, you can install a custom version using [uv](https://docs.astral.sh/uv/getting-started/installation/) tool:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv -p 3.10
source .venv/bin/activate
# optionally you might need to install pip which is not installed by default:
uv pip install -U pip
```

## CloudAI Modes Usage Examples

CloudAI supports five modes:
- [install](install) - Use the install mode to install all test templates in the specified installation path
- [dry-run](dry-run) - Use the dry-run mode to simulate running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups
- [run](run) - Use the run mode to run experiments
- [generate-report](generate-report) - Use the generate-report mode to generate reports under the test directories alongside the raw data
- [uninstall](uninstall) - Use the uninstall mode to remove installed test templates

(install)=
### install

To install test prerequisites, run CloudAI CLI in install mode.

Please make sure to use the correct system configuration file that corresponds to your current setup for installation and experiments.
```bash
cloudai install\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test
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
In the generate-report mode, use the --result-dir argument to specify a subdirectory under the output directory.
This subdirectory is usually named with a timestamp for unique identification.
(uninstall)=
### uninstall
To uninstall test prerequisites, run CloudAI CLI in uninstall mode:
```bash
cloudai uninstall\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test
```
Verify TOML configs:
```bash
# verify all at once
cloudai verify-configs conf

# verify a single file
cloudai verify-configs conf/common/system/example_slurm_cluster.toml

#  verify all scenarios using specific folder with Test TOMLs
cloudai verify-configs --tests-dir conf/release/spcx/l40s/test conf/release/spcx/l40s/test_scenario
```

## Contribution
Please feel free to contribute to the CloudAI project and share your insights. Your contributions are highly appreciated.

## License
This project is licensed under Apache 2.0. See the LICENSE file for detailed information.

## Additional Documentation
For more detailed instructions and guidance, including advanced usage and troubleshooting, please refer to the [USER_GUIDE.md](https://github.com/NVIDIA/cloudai/blob/main/USER_GUIDE.md)

```{toctree}
:maxdepth: 2
:caption: Contents:

DEV
ai_dynamo
reporting
```
