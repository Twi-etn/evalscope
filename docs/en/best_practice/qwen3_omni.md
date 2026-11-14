# Best Practices for Evaluating the Qwen3-Omni Model

This guide shows how to serve Qwen3-Omni with vLLM, then benchmark inference throughput and latency with EvalScope.

## Dependency Installation

First, install the [EvalScope](https://github.com/modelscope/evalscope) model evaluation framework:

```bash
pip install 'evalscope[app,perf]' -U
```

## Model Service Inference Performance Evaluation

For multimodal models, we need to access model capabilities through an OpenAI API-compatible inference service for evaluation, as local usage of transformers for inference is not yet supported. You can deploy the model to a cloud service supporting the OpenAI API interface or choose to start the model locally using frameworks like vLLM or ollama. These inference frameworks effectively support multiple concurrent requests, accelerating the evaluation process.

Below, we demonstrate how to start the Qwen3-Omni-30B-A3B-Instruct model service locally using vLLM and conduct performance evaluation.

### Dependency Installation

Since the code submitted to vLLM is currently (as of September 25, 2025) in the pull request stage, and audio output inference support for the instruction model will be released soon, you can install vLLM from the source using the following commands. Note that we recommend creating a new Python environment to avoid runtime conflicts and incompatibilities:

```shell
git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
cd vllm
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation
# If you encounter an "Undefined symbol" error while using VLLM_USE_PRECOMPILED=1, please use "pip install -e . -v" to build from source.
# Install the Transformers
pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install qwen-omni-utils -U
pip install -U flash-attn --no-build-isolation
```

### Starting the Model Service

The following command can be used to create an API endpoint at `http://localhost:8801/v1`:

```bash
VLLM_USE_MODELSCOPE=true vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8801 --dtype bfloat16 --max-model-len 32768 --served-model-name Qwen3-Omni-30B-A3B-Instruct
```

### Local Model Service Performance Evaluation

**Stress Test Command:**

- Test Environment: A100 80G * 1
- Input: 1024 tokens of text + 1 image of 512x512
- Output: 1024 tokens

> Note: The tokenizer path of Qwen3-30B-A3B-Instruct is used here because Qwen3-Omni-30B-A3B-Instruct and Qwen3-30B-A3B-Instruct use the same text tokenizer.

```shell
evalscope perf \
    --model Qwen3-Omni-30B-A3B-Instruct \
    --url http://localhost:8801/v1/chat/completions \
    --api-key "API_KEY" \
    --parallel 1 5 10 50 100 \
    --number 2 10 20 100 200 \
    --api openai \
    --dataset random_vl \
    --min-tokens 1024 \
    --max-tokens 1024 \
    --prefix-length 0 \
    --min-prompt-length 1024 \
    --max-prompt-length 1024 \
    --image-width 512 \
    --image-height 512 \
    --image-format RGB \
    --image-num 1 \
    --tokenizer-path Qwen/Qwen3-30B-A3B-Instruct-2507
```

The output report is as follows:

![image](https://sail-moe.oss-cn-hangzhou.aliyuncs.com/yunlin/images/evalscope/doc/qwen_omni/eval.png)

## Visualization

Run `evalscope service` and open the **Performance** page to inspect request latency, throughput and token usage for the completed run.
