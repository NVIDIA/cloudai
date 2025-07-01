## Running AI Dynamo with CloudAI
CloudAI supports end-to-end inference benchmarking of large language models using [AI Dynamo](https://github.com/ai-dynamo/dynamo). This section explains how to run AI Dynamo jobs via CloudAI, beginning with setting up the environment and downloading Hugging Face model weights, and continuing through job submission and monitoring.

In particular, this section will cover:

- How to download model weights using `huggingface-cli` and configure `HUGGING_FACE_HOME`
- How to write and adjust a CloudAI test schema for AI Dynamo
- How to switch the model or scale node resources
- How to monitor the job and interpret the results

CloudAI abstracts away most of the complexity in coordinating frontend, prefill, and decode nodes for AI Dynamo. Users are responsible for downloading the model weights, configuring the appropriate environment variable, and preparing the test schema.

---

### Step 1: Download Model Weights Using Hugging Face CLI

Install the Hugging Face CLI:

```bash
$ pip install -U "huggingface_hub[cli]"
```

Log in using your Hugging Face token:

```bash
$ huggingface-cli login
```

Download the model weights and tokenizer to a HF_HOME that will serve as the Hugging Face cache:

```bash
$ export HF_HOME=/path/to/hf_home/
$ huggingface-cli download nvidia/Llama-3.1-405B-Instruct-FP8
$ huggingface-cli download hf-internal-testing/llama-tokenizer
```

You can verify the model cache using:

```bash
$ huggingface-cli scan-cache -vvv

REPO ID                             REPO TYPE REVISION                                 SIZE ON DISK NB FILES LAST_MODIFIED REFS LOCAL PATH
----------------------------------- --------- ---------------------------------------- ------------ -------- ------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------
hf-internal-testing/llama-tokenizer model     d02ad6cb9dd2c2296a6332199fa2fdca5938fef0         2.3M        5 3 days ago    main /path/to/hf_home/hub/models--hf-internal-testing--llama-tokenizer/snapshots/d02ad6cb9dd2c2296a6332199fa2fdca5938fef0
nvidia/Llama-3.1-405B-Instruct-FP8  model     a0a0bc4e698fbbe4eb184bbd62067ff195a65a39       410.1G       96 4 days ago    main /path/to/hf_home/hub/models--nvidia--Llama-3.1-405B-Instruct-FP8/snapshots/a0a0bc4e698fbbe4eb184bbd62067ff195a65a39

Done in 0.3s. Scanned 2 repo(s) for a total of 410.1G.
```

The path to the downloaded weights should be consistent with the structure expected by the Hugging Face ecosystem.

---

### Step 2: Configure `HF_HOME` in the Test Schema

Set the `HF_HOME` environment variable in the test schema file (e.g., `test.toml`) so that CloudAI can locate the model weights:

```toml
name = "llama3.1_405b_fp8"
description = "llama3.1_405b_fp8"
test_template_name = "AIDynamo"

[cmd_args]
docker_image_url = "/path/to/docker/image"
served_model_name = "nvidia/Llama-3.1-405B-Instruct-FP8"

  [cmd_args.dynamo.processor]
  [cmd_args.dynamo.router]
  [cmd_args.dynamo.frontend]
  [cmd_args.dynamo.prefill_worker]
  num_nodes = 1

  [cmd_args.dynamo.vllm_worker]
  num_nodes = 0

  [cmd_args.genai_perf]
  endpoint = "v1/chat/completions"
  endpoint_type = "chat"
  streaming = true

[extra_env_vars]
HF_HOME = "/your/path/to/hf_home"
```

This environment variable should point to the root directory used with `--local-dir` in the download step. CloudAI will use this directory to locate and load the appropriate model weights.

---

### Step 3: Node Configuration for AI Dynamo

AI Dynamo jobs use three distinct types of nodes:

- **Frontend node**: Hosts the coordination services (`etcd`, `nats`) as well as the **frontend server** and the **request generator** (`genai-perf`)
- **Prefill node(s)**: Handle the prefill stage of inference
- **Decode node(s)**: Handle the decode stage of inference (optional, depending on model and setup)

The total number of nodes required must be:

```
1 (frontend) + num_prefill_nodes + num_decode_nodes
```

If there is a mismatch in the number of nodes between the schema and the test scenario, CloudAI will use the number of nodes specified in the test schema, ignoring the value in the test scenario.

All node role assignments and orchestration are automatically managed by CloudAI.

---

### Step 4: Launching and Monitoring the Job

To run the job:

```bash
$ python cloudaix.py run --system-config conf/staging/ai_dynamo/system/oci.toml --tests-dir conf/staging/ai_dynamo/test --test-scenario conf/staging/ai_dynamo/test_scenario/ai_dynamo.toml
```

#### Option 1: Monitor via Slurm

```bash
$ watch squeue --me
```

#### Option 2: Monitor Output Logs

Navigate to the results directory created by CloudAI and observe the logs:

```bash
$ cd ./results/../
$ watch tail -n 4 *.txt
```

The frontend node will initially wait to allow weight loading on all nodes. Once ready, it will launch `genai-perf`, which begins generating requests to the frontend server. All servers cooperate to complete inference, and the output will appear in `stdout.txt`.

### Step 5: Review Results
After job completion, CloudAI will place the output logs and result files in the designated results directory. To analyze performance metrics and validate inference outcomes:

- Navigate to the results directory (e.g., ./results/...)
- Most importantly, open the profile_genai_perf.csv file to examine the final benchmarking results

This CSV file includes detailed metrics collected by genai-perf, such as request latency, throughput, and system utilization statistics. Use this data to evaluate the model's performance and identify potential bottlenecks or optimization opportunities.

```
Metric,avg,min,max,p99,p95,p90,p75,p50,p25
Time To First Token (ms),"1,146.31",249.48,"3,485.23","3,457.97","3,349.56","3,215.06","1,330.93",640.07,286.52
Time To Second Token (ms),26.05,0.00,133.51,96.12,36.56,34.88,34.35,33.55,1.78
Request Latency (ms),"6,406.20","5,371.47","9,608.72","9,436.13","9,046.58","9,028.16","6,549.60","5,690.23","5,493.63"
Inter Token Latency (ms),30.35,27.59,35.60,35.23,33.88,32.53,31.05,30.13,29.04
Output Sequence Length (tokens),174.45,164.00,187.00,186.22,183.10,180.10,177.00,174.00,171.75
Input Sequence Length (tokens),"3,000.05","2,999.00","3,001.00","3,001.00","3,001.00","3,000.00","3,000.00","3,000.00","3,000.00"

Metric,Value
Output Token Throughput (per sec),261.25
Request Throughput (per sec),1.50
Request Count (count),40.00
```
