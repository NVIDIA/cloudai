CI tester
# CloudAI Benchmark Framework

## Project Description
CloudAI benchmark framework aims to develop an industry standard benchmark focused on grading Data Center (DC) scale AI systems in the Cloud. The primary motivation is to provide automated benchmarking on various systems.

## Key Concepts
### Schemas
CloudAI operates on four main schemas:

1. **System Schema**: Describes the system, including the scheduler type, node list, and global environment variables.
2. **Test Template Schema**: A template for tests that includes all required command-line arguments and environment variables. This schema allows users to separate test template implementations from systems.
3. **Test Schema**: An instance of a test template with custom arguments and environment variables.
4. **Test Scenario Schema**: A set of tests with dependencies and additional descriptions about the test scenario.

These schemas enable CloudAI to be flexible and compatible with different systems and configurations.


## Support matrix
|Test|Slurm|Kubernetes (experimental)|Standalone|
|---|---|---|---|
|ChakraReplay|✅|❌|❌|
|GPT|✅|❌|❌|
|Grok|✅|❌|❌|
|NCCL|✅|✅|❌|
|NeMo Launcher|✅|❌|❌|
|Nemotron|✅|❌|❌|
|Sleep|✅|✅|✅|
|UCC|✅|❌|❌|


## Set Up Access to the Private NGC Registry
First, ensure you have access to the Docker repository. Follow these steps:

1. **Sign In**: Go to [NVIDIA NGC](https://ngc.nvidia.com/signin) and sign in with your credentials.
2. **Generate API Key**:
    - On the top right corner, click on the dropdown menu next to your profile.
    - Select "Setup".
    - In the "Setup" section, find "Keys/Secrets".
    - Click "Generate API Key" and confirm when prompted. A new API key will be presented.
    - **Important**: Save this API key locally as you will not be able to view it again on NGC.

Next, set up your enroot credentials. Ensure you have the correct credentials under `~/.config/enroot/.credentials`:
```
machine nvcr.io login $oauthtoken password <api-key>
```
- Replace `<api-key>` with your respective credentials. Keep `$oauthtoken` as is.


## Quick Start
Clone the CloudAI repository to your local machine:
```bash
git clone git@github.com:NVIDIA/cloudai.git
cd cloudai
```

Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

Next, install the required packages using pip:
```bash
pip install -r requirements.txt
```

After setting up the environment and installing dependencies, install the cloudai package itself:
```bash
pip install .
```

CloudAI supports five modes: install, dry-run, run, generate-report, and uninstall.
* Use the install mode to install all test templates in the specified installation path.
* Use the dry-run mode to simulate running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups.
* Use the run mode to run experiments.
* Use the generate-report mode to generate reports under the test directories alongside the raw data.
* Use the uninstall mode to remove installed test templates.

To install test prerequisites, run CloudAI CLI in install mode.
Please make sure to use the correct system configuration file that corresponds to your current setup for installation and experiments.
```bash
cloudai install\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test
```

To simulate running experiments without execution, use the dry-run mode:
```bash
cloudai dry-run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

To run experiments, execute CloudAI CLI in run mode:
```bash
cloudai run\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --test-scenario conf/common/test_scenario/sleep.toml
```

To generate reports, execute CloudAI CLI in generate-report mode:
```bash
cloudai generate-report\
    --system-config conf/common/system/example_slurm_cluster.toml\
    --tests-dir conf/common/test\
    --result-dir /path/to/result_directory
```
In the generate-report mode, use the --result-dir argument to specify a subdirectory under the output directory.
This subdirectory is usually named with a timestamp for unique identification.

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

## Contributing
Feel free to contribute to the CloudAI project. Your contributions are highly appreciated.

## License
This project is licensed under Apache 2.0. See the LICENSE file for detailed information.

## Additional Documentation
For more detailed instructions and guidance, including advanced usage and troubleshooting, please refer to the [USER_GUIDE.md](./USER_GUIDE.md)
