# Quick Start

EvalScope is an evaluation framework designed for large models, aiming to provide a simple, easy-to-use, and comprehensive evaluation process. This guide will walk you through various evaluation tasks from simple to complex, helping you get started quickly.

## Start Your First Evaluation

Prepare the GSM8K evaluation split at `./data/gsm8k/main_test.parquet` and its training split for few-shot examples at `./data/gsm8k/main_train.parquet`. You can then run the evaluation through the command line or Python.

### Method 1: Using Command Line

Execute the `evalscope eval` command in any path to start evaluation. This is the most recommended way to get started.

```bash
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k \
 --dataset-args '{"gsm8k":{"local_path":"./data/gsm8k"}}' \
 --limit 5
```

**Basic Parameter Description:**
- `--model`: Specify the model's ModelScope ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) or local path.
- `--datasets`: Specify one or more dataset names, separated by spaces. For supported datasets, please refer to [Dataset List](./supported_dataset/index.md).
- `--limit`: Maximum number of samples to evaluate per dataset, convenient for quick verification. If not set, all data will be evaluated.

### Method 2: Using Python Code

By calling the `run_task` function with a `TaskConfig` configuration, you can run evaluation in a Python environment. The configuration can be a `TaskConfig` object, Python dictionary, or `yaml`/`json` file path.

::::{tab-set}
:::{tab-item} Using `TaskConfig` Object
```python
from evalscope.run import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-0.5B-Instruct',
    datasets=['gsm8k'],
    dataset_args={'gsm8k': {'local_path': './data/gsm8k'}},
    limit=5
)

run_task(task_cfg=task_cfg)
```
:::
:::{tab-item} Using Python Dictionary
```python
from evalscope.run import run_task

task_cfg = {
    'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'datasets': ['gsm8k'],
    'dataset_args': {'gsm8k': {'local_path': './data/gsm8k'}},
    'limit': 5
}

run_task(task_cfg=task_cfg)
```
:::
:::{tab-item} Using YAML File
```{code-block} yaml
:caption: config.yaml

model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - gsm8k
dataset_args:
  gsm8k:
    local_path: ./data/gsm8k
limit: 5
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.yaml")
```
:::
:::{tab-item} Using JSON File
```{code-block} json
:caption: config.json

{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "datasets": ["gsm8k"],
    "dataset_args": {"gsm8k": {"local_path": "./data/gsm8k"}},
    "limit": 5
}
```
```python
from evalscope.run import run_task

run_task(task_cfg="config.json")
```
:::
::::

### View Evaluation Results

Native evaluation saves one Parquet file per model, benchmark, and subset under each run's `predictions/` and `reviews/` directories. Each row represents a sample. Fields with flexible nested shapes, such as `messages`, `model_output`, and `sample_score`, are stored as JSON text columns. Benchmark aggregates remain JSON under `reports/`.

Use PyArrow or Pandas to query the columns directly, or use `read_result_rows` to expand JSON fields into Python objects. The Dashboard and cache resume read the Parquet results through this same function.

```python
from evalscope.api.evaluator.result_io import read_result_rows

rows = read_result_rows('outputs/<run>/reviews/<model>/<benchmark>_<subset>.parquet')
```

After evaluation is complete, the terminal will print a score report in the following format:

```text
┌───────────────────────┬───────────┬────────────┬───────────────┬───────┬─────────┐
│ Model                 │ Dataset   │ Metric     │ Subset        │   Num │ Score   │
├───────────────────────┼───────────┼────────────┼───────────────┼───────┼─────────┤
│ Qwen2.5-0.5B-Instruct │ gsm8k     │ Accuracy ↑ │ main          │     5 │ 40%     │
└───────────────────────┴───────────┴────────────┴───────────────┴───────┴─────────┘
```

```{seealso}
In addition to command line reports, you can also use our visualization tools to analyze evaluation results in depth.

Reference: [Evaluation Result Visualization](visualization.md)
```

## Advanced Usage

### Customize Evaluation Parameters

 **Example 1. Adjusting Model Loading and Inference Generation Parameters**

You can fine-tune the control of model loading, inference generation, and dataset configuration by passing JSON-formatted strings.

```shell
evalscope eval \
 --model Qwen/Qwen3-0.6B \
 --model-args '{"revision": "master", "precision": "torch.float16", "device_map": "auto"}' \
 --generation-config '{"do_sample":true,"temperature":0.6,"max_tokens":512}' \
 --dataset-args '{"gsm8k": {"local_path": "./data/gsm8k", "few_shot_num": 0, "few_shot_random": false}}' \
 --datasets gsm8k \
 --limit 10
```

**Common Parameter Descriptions:**
- `--model-args`: Model loading parameters, such as `revision` (version), `precision` (precision), `device_map` (device mapping).
- `--generation-config`: Inference generation parameters, such as `do_sample` (sampling), `temperature` (temperature), `max_tokens` (maximum length).
- `--dataset-args`: Dataset-specific parameters, keyed by dataset name. For example, `few_shot_num` (number of few-shot examples).

 **Example 2. Adjusting Result Aggregation Methods**

For tasks that require multiple generations (such as math problems), you can specify the number of repeated generations k using the `--repeats` parameter, and adjust the result aggregation method through `dataset-args`, such as `mean_and_vote_at_k`, `mean_and_pass_at_k`, `mean_and_pass^k`.

```shell
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k \
 --limit 10 \
 --dataset-args '{"gsm8k": {"local_path": "./data/gsm8k", "aggregation": "mean_and_vote_at_k"}}' \
 --repeats 5
```

```{seealso}
To view all configurable options, please refer to: [Complete Parameter Description](parameters.md)
```

### Evaluate Online Model APIs

EvalScope supports evaluating model services compatible with OpenAI API format. Simply specify the service address, API Key, and set `eval-type` to `openai_api`.

For System One decision models, use `--eval-type systemone_api`. See [System One Evaluation](../user_guides/systemone.md).

**1. Start Model Service**

Using vLLM as an example, start a model service:
```shell
# Please install vLLM first: pip install vllm
export VLLM_USE_MODELSCOPE=True
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen2.5 \
  --trust-remote-code \
  --port 8801
```

**2. Run Evaluation**

Use the following command to evaluate the API service:
```shell
evalscope eval \
 --model qwen2.5 \
 --api-url http://127.0.0.1:8801/v1 \
 --api-key EMPTY \
 --eval-type openai_api \
 --datasets gsm8k \
 --dataset-args '{"gsm8k":{"local_path":"./data/gsm8k"}}' \
 --limit 10
```

For OpenAI's official Responses API, use `--eval-type openai_responses_api` and point `--api-url` at the API root:
```shell
evalscope eval \
 --model gpt-4.1-mini \
 --api-url https://api.openai.com/v1 \
 --api-key $OPENAI_API_KEY \
 --eval-type openai_responses_api \
 --datasets gsm8k \
 --dataset-args '{"gsm8k":{"local_path":"./data/gsm8k"}}' \
 --limit 10 \
 --generation-config '{"stream": true, "max_tokens": 512}'
```

### Using Judge Models for Assessment

For certain subjective or open-ended tasks (such as `simple_qa`), you can specify a powerful "judge model" to evaluate the target model's output.

```python
import os
from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model='qwen2.5-7b-instruct',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    eval_type='openai_api',
    datasets=['simple_qa'],
    limit=5,
    judge={
        'strategy': 'auto',
        'models': {
            'model_id': 'qwen2.5-72b-instruct',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': os.getenv('DASHSCOPE_API_KEY'),
        },
    }
)

run_task(task_cfg=task_cfg)
```

```{seealso}
For detailed configuration of judge models, please refer to: [Judge Model Parameters](./parameters.md#judge-parameters)
```

### Load Local Datasets

Native evaluation reads local datasets. Pass a filesystem path through `dataset_args.<benchmark>.local_path`.

The standard loader accepts a Parquet file, a Lance dataset directory, or a parent directory containing named splits. In a parent directory it looks for `<subset>_<split>.parquet` or `<subset>_<split>.lance`, then `<subset>.parquet` or `<subset>.lance`. Specialized benchmarks may require their own directory layout and associated media.

Install the optional Lance dependency from the repository:

```bash
uv pip install -e '.[lance]'
```

```bash
evalscope eval \
 --model local-test --eval-type mock_llm \
 --datasets general_mcq \
 --dataset-args '{"general_mcq":{"local_path":"./data/questions.lance","dataset_revision":1,"few_shot_num":0}}' \
 --limit 5
```

Omit `dataset_revision` to read the latest Lance version. You can also pass a version number or a tag. Standard Lance loaders record the concrete versions used by the run. To resume after a new version is published, use the same pinned version and configuration as the original run.

For CSV, TSV, or JSONL inputs, convert each split to Parquet or Lance before evaluation. For example, this converts the repository's JSONL validation fixture:

```bash
python -c "from pyarrow import json, parquet; parquet.write_table(json.read_json('custom_eval/text/mcq/example_val.jsonl'), 'example_val.parquet')"
```


### Offline Evaluation

In offline environments, you can use locally cached models and datasets for evaluation.

**1. Prepare Local Datasets**

Prepare the benchmark data locally before starting an evaluation. For example, you can download the upstream files from ModelScope in advance:

1) First, check the ModelScope ID of the dataset you want to use: Find the **Dataset ID** of the dataset you need in the [Supported Datasets](./supported_dataset/index.md) list, for example, the ID for `mmlu_pro` is `modelscope/MMLU-Pro`.
2) Use the modelscope command to download the dataset: Click "Dataset Files" tab -> Click "Download Dataset" -> Copy the command line
```bash
# Download dataset
modelscope download --dataset modelscope/MMLU-Pro --local_dir ./data/mmlu_pro
```
3) Convert or rename the downloaded splits to the local layout described above. Pass their parent directory as `local_path`.

**2. Prepare Local Models**

Model files are hosted on the ModelScope Hub and require internet connection to load. When you need to create evaluation tasks in an offline environment, you can download models locally in advance:

For example, use the modelscope command to download the [Qwen2.5-0.5B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct) model locally:
```bash
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./model/qwen2.5
```

```{seealso}
For more download options, please refer to [ModelScope Download Guide](https://modelscope.cn/docs/models/download).
```

**3. Run Offline Evaluation**

In the evaluation command, point `--model` to the local model path and specify the dataset's `local_path` through `--dataset-args`.

```shell
evalscope eval \
 --model ./model/qwen2.5 \
 --datasets mmlu_pro \
 --dataset-args '{"mmlu_pro": {"local_path": "./data/mmlu_pro"}}' \
 --limit 10
```

### Continue Evaluation from Existing/Interrupted Results

For Native evaluations, resume an interrupted run by repeating the original command with `--use-cache` set to its output directory. Use the directory containing `predictions/`, `reviews/`, `reports/`, and `configs/`, such as `outputs/20261009_120000`, rather than one of those subdirectories. You can find this path in the original run's logs.

For example, if the following evaluation was interrupted:

```shell
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k \
 --dataset-args '{"gsm8k":{"local_path":"./data/gsm8k"}}' \
 --limit 10
```

Resume it using the actual output directory from that run:

```shell
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k \
 --dataset-args '{"gsm8k":{"local_path":"./data/gsm8k"}}' \
 --limit 10 \
 --use-cache outputs/20261009_120000
```

EvalScope matches cached results by sample ID:

- Samples with both a cached prediction and review reuse their scores.
- Samples with a cached prediction but no review run only the scoring step.
- Samples without a valid cached prediction run inference and scoring.

The run writes remaining results and regenerates reports in the same output directory. Keep the original model, dataset selection, prompts, generation settings, and scoring configuration when continuing the same evaluation. Native evaluations validate the cached evaluation identity before reusing results; incompatible settings are rejected.

To deliberately re-score existing predictions, add `--rerun-review` to the resume command. This recomputes reviews instead of reusing cached scores and atomically replaces the review cache after success. Samples without cached predictions still require inference. `--rerun-review` also explicitly allows an evaluation identity mismatch, so use it only when those cached predictions are appropriate for the new scoring configuration.

### Skip Samples with Evaluation Errors

In some cases, the model may fail to infer due to overly long inputs, timeouts, model service crashes, or other reasons, or unknown errors may occur during the evaluation process. In such cases, the program will throw an exception and interrupt the evaluation flow. You can use the `--ignore-errors` parameter to skip these error samples and continue evaluating the remaining ones. In the evaluation report, these samples will be ignored, meaning the final number of evaluated samples will be reduced.

```shell
evalscope eval \
 --model Qwen/Qwen2.5-0.5B-Instruct \
 --datasets gsm8k \
 --dataset-args '{"gsm8k":{"local_path":"./data/gsm8k"}}' \
 --limit 10 \
 --ignore-errors
```

## Migration from v0.1.x

If you previously used v0.1.x version, please note the following major changes after upgrading to v1.0+:

1.  **Local Datasets**: `zip` compressed package format is no longer supported. Please refer to the [Offline Evaluation](#offline-evaluation) guide and use the `local_path` parameter to load local datasets.
2.  **Visualization Compatibility**: The output report format in v1.0 has been updated, and old reports are incompatible with the new visualization tools.
3.  **Inference Parameters**: The model inference parameter `n` has been removed. Please use the `repeats` parameter to control the number of repeated generations.
4.  **Evaluation Process Control**: The `stage` parameter has been removed. Added `rerun_review` parameter to force re-execution of the scoring step when `use_cache=True`.
5.  **Dataset Names**: The `gpqa` dataset has been renamed to `gpqa_diamond` and no longer requires manual specification of `subset_list`.

## Frequently Asked Questions

Encountering issues during evaluation? We've compiled a list of frequently asked questions to help you solve problems.

```{seealso}
Reference: [Frequently Asked Questions](faq.md)
```
