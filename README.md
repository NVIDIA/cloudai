# CloudAI Benchmark Suite

## Project Description
CloudAI benchmark suite aims to develop an industry standard benchmark focused on grading Data Center (DC) scale AI systems in the Cloud.

## Quick Start
First, clone the CloudAI repository to your local machine:
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

To install test templates, run CloudAI CLI in install mode.
Please make sure to use the correct system configuration file that corresponds to your current setup for installation and experiments.
```bash
cloudai\
    --mode install\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml
```

To simulate running experiments without execution, use the dry-run mode:
```bash
cloudai\
    --mode dry-run\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml\
    --test_scenario_path conf/v0.6/general/test_scenario/sleep.toml
```

To run experiments, execute CloudAI CLI in run mode:
```bash
cloudai\
    --mode run\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml\
    --test_scenario_path conf/v0.6/general/test_scenario/sleep.toml
```

To generate reports, execute CloudAI CLI in generate-report mode:
```bash
cloudai\
    --mode generate-report\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml\
    --output_path /path/to/output_directory
```
In the generate-report mode, use the --output_path argument to specify a subdirectory under the result directory.
This subdirectory is usually named with a timestamp for unique identification.

To uninstall test templates, run CloudAI CLI in uninstall mode:
```bash
cloudai\
    --mode uninstall\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml
```

# Contributing
Feel free to contribute to the CloudAI project. Your contributions are highly appreciated.

# License
This project is licensed under Apache 2.0. See the LICENSE file for detailed information.
