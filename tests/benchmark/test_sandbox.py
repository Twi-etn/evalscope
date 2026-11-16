# Copyright (c) Alibaba, Inc. and its affiliates.
from dotenv import dotenv_values

env = dotenv_values('.env')

from evalscope.constants import EvalType, JudgeStrategy
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestCodeBenchmark(TestBenchmark):
    """Benchmark evaluation test cases."""

    def setUp(self):
        """Setup common test configuration."""
        self.base_config = {
            'model': 'qwen3.8-flash',
            'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': env.get('DASHSCOPE_API_KEY'),
            'eval_type': EvalType.OPENAI_API,
            'eval_batch_size': 5,
            'limit': 5,
            'generation_config': {
                'max_tokens': 4096,
                'temperature': 0.0,
                'seed': 42,
                'parallel_tool_calls': True,
                'stream': True,
            },
            'judge': {
                'strategy': JudgeStrategy.AUTO,
                'models': {
                'model_id': 'qwen3.8-flash',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'max_tokens': 4096,
                }
                },
            },
            'use_sandbox': True,
            'sandbox_type': 'docker',
            'debug': True,
        }

    def test_terminal_bench_v2(self):
        """Test Terminal-Bench v2 dataset."""
        dataset_args = {
            'extra_params': {
                # 'agent_name': 'oracle',
                'timeout_multiplier': 3
            },
        }
        self._run_dataset_test('terminal_bench_v2', dataset_args, limit=5, use_sandbox=False, rerun_review=True, model='qwen3-coder-plus', eval_batch_size=2)

    def test_terminal_bench_v2_1(self):
        """Test Terminal-Bench v2.1 dataset with environment_kwargs."""
        dataset_args = {
            'extra_params': {
                'timeout_multiplier': 3,
                'environment_kwargs': {'override_cpus': 2},
            },
        }
        self._run_dataset_test('terminal_bench_v2_1', dataset_args, limit=5, use_sandbox=False, rerun_review=True, model='qwen3-coder-plus', eval_batch_size=2)
