# OSU

This workload (`test_template_name` is `OSUBench`) allows you to execute OSU Micro Benchmarks
within the CloudAI framework.

## Usage example

Test example:

``` toml
    name = "osu_example"
    test_template_name = "OSUBench"
    description = "OSU Benchmark example"

    [cmd_args]
    "docker_image_url" = "docker-image-with-osu-benchmark:latest"
    "location" = "/directory/with/osu/binaries/in/container"
    "benchmark" = ["osu_allreduce", "osu_allgather"]
    "iterations" = 10
    "message_size" = "1024"
```

Test Scenario example:

``` toml
    name = "osu_example"

    [[Tests]]
    id = "Tests.1"
    test_name = "osu_example"
    num_nodes = "2"
    time_limit = "00:20:00"
```

## Arguments

In the table below you can find arguments that can be used in `[cmd_args]`
section:

| Argument           | Type                    | Required | CLI flag | Description                                                       |
|--------------------|-------------------------|----------|----------|-------------------------------------------------------------------|
| docker_image_url   | string                  | yes      | —        | URL of the Docker image to use for the test.                      |
| location           | string                  | yes      | —        | Path inside the container to the OSU Benchmark binaries.          |
| benchmark          | string or list[string]  | yes      | —        | Benchmark name(s) to run (e.g., `osu_allreduce`, `osu_allgather`).|
| message_size       | string or list[string]  | no       | -m       | Message size or range. Examples: `128`, `2:128`, `2:`.            |
| iterations         | int                     | no       | -i       | Number of iterations to run.                                      |
| warmup             | int                     | no       | -x       | Number of warmup iterations to skip before timing.                |
| mem_limit          | int (bytes)             | no       | -M       | Per-process maximum memory consumption in bytes.                  |

Notes:
- `benchmark` accepts a single value or a list of benchmarks.
- `message_size` examples:
  - `-m 128` sets max to 128 with default min
  - `-m 2:128` sets min 2 and max 128
  - `-m 2:` sets min 2 with default max
