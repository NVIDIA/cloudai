The functional test use the dry-run mode and compare the output with an expected output.

### The test
The test receives two parameters:
- The path to the scenario that need to be executed
- The path to the expected output directory, this directory should be the same as the result directory written by the dry-run execution of the scenario

It execute the scenario in dry-run mode and compare the results folder with the expected output folder, if any difference is found in one of the files, the test fails.
**Note that the empty lines and commented lines (the ones that start with #) are not taken into account when computing the difference.**

### The workflow
The workflow uses a strategy matrix to define the tests it should run, every tests run in parallel.
Note that the python setup is cached and therefore will run only if the requirements-dev.txt file is changed.

### Add a new test
To add a new test:
- Create the desired scenario
- Create a folder under `ci_tools/functional_tests/scenarios_expected_outputs/` named after the name of this test and fill it with the expected output of the scenario (you can copy the output of a valid dry run)
- In the pipeline file (`.github/workflows/functional_test.yml`), add your new test in `jobs.test.strategy.matrix.include` (see the comment saying "Add your new test here"), follow the syntax of the other tests

