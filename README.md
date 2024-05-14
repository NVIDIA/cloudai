# Cloud AI Benchmark Suite

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

Cloud AI supports five modes: install, dry-run, run, generate-report, and uninstall.
* Use the install mode to install all test templates in the specified installation path.
* Use the dry-run mode to simulate running experiments without actually executing them. This is useful for verifying configurations and testing experiment setups.
* Use the run mode to run experiments.
* Use the generate-report mode to generate reports under the test directories alongside the raw data.
* Use the uninstall mode to remove installed test templates.

To install test templates, run Cloud AI CLI in install mode.
Please make sure to use the correct system configuration file that corresponds to your current setup for installation and experiments.
```bash
python -m cloudai\
    --mode install\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml
```

To simulate running experiments without execution, use the dry-run mode:
```bash
python -m cloudai\
    --mode dry-run\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml\
    --test_scenario_path conf/v0.6/general/test_scenario/sleep/test_scenario.toml
```

To run experiments, execute Cloud AI CLI in run mode:
```bash
python -m cloudai\
    --mode run\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml\
    --test_scenario_path conf/v0.6/general/test_scenario/sleep/test_scenario.toml
```

To generate reports, execute Cloud AI CLI in generate-report mode:
```bash
python -m cloudai\
    --mode generate-report\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml\
    --output_path /path/to/output_directory
```
In the generate-report mode, use the --output_path argument to specify a subdirectory under the result directory.
This subdirectory is usually named with a timestamp for unique identification.

To uninstall test templates, run Cloud AI CLI in uninstall mode:
```bash
python -m cloudai\
    --mode uninstall\
    --system_config_path conf/v0.6/general/system/example_slurm_cluster.toml
```

# Implementing a custom execution flow

```py
# Create a System, TestTemplate and TestScenario objects
system, test_templates, test_scenario = System(), [TestTemplate()], TestScenario()

# One way to do that is to use the Parser class
parser = Parser(<system_config_path>, <test_template_path>, <test_path>, <test_scenario_path>)
system, test_templates, test_scenario = parser.parse()

# Update the system object using relevant SystemObjectUpdater. This is necessary to update the system object
# with the correct system configuration.
system_object_updater = SystemObjectUpdater()
system_object_updater.update(system)

# Check if test templates are installed
installer = Installer(system)
if not installer.is_installed(test_templates):
    # raise an exception or
    installer.install(test_templates)

# Create a Runner object and run the test scenario. Use asyncio.run() to run the async function.
runner = Runner(args.mode, system, test_scenario)
asyncio.run(runner.run())

# Generate a report...
generator = ReportGenerator(runner.runner.output_path)
generator.generate_report(test_scenario)

# ... and grade the test scenario
grader = Grader(runner.runner.output_path)
grader.grade(test_scenario)
```

# Contributing
Feel free to contribute to the CloudAI project. Your contributions are highly appreciated.

# License
This project is licensed under Apache 2.0. See the LICENSE file for detailed information.
