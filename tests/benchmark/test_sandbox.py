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

    def test_mbpp(self):
        """Test MBPP dataset."""
        dataset_args = {
            # 'metric_list': ['Pass@1']
        }
        self._run_dataset_test('mbpp', dataset_args, limit=20, debug=False)

    def test_multipl_e_mbpp(self):
        """Test MultiPL-E MBPP dataset."""
        dataset_args = {
            'subset_list': [
                'mbpp-cpp', # ok
                'mbpp-ts', # ok
                'mbpp-sh', # ok
                # 'mbpp-cs', # need x86_64 docker image
                'mbpp-go', # ok
                'mbpp-java', # ok
                'mbpp-lua', # ok
                'mbpp-js', # ok
                'mbpp-php', # ok
                'mbpp-pl', # ok
                'mbpp-rkt', # ok
                'mbpp-r', # ok
                'mbpp-rs', # ok
                'mbpp-scala', # ok
                'mbpp-swift', # ok
                'mbpp-rb', # ok
                'mbpp-d', # ok
                'mbpp-jl', # ok
        ],
        }
        self._run_dataset_test('multiple_mbpp', dataset_args, limit=10, use_cache='outputs/20251210_150606', rerun_review=True, debug=False, eval_batch_size=2)

    def test_multipl_e_humaneval(self):
        """Test MultiPL-E HumanEval dataset."""
        dataset_args = {
            'subset_list': [
                'humaneval-cpp', # ok
                'humaneval-ts', # ok
                'humaneval-sh', # ok
                'humaneval-cs', # need x86_64 docker image
                'humaneval-go', # ok
                'humaneval-java', # ok
                'humaneval-lua', # ok
                'humaneval-js', # ok
                'humaneval-php', # ok
                'humaneval-pl', # ok
                'humaneval-rkt', # ok
                'humaneval-r', # ok
                'humaneval-rs', # ok
                'humaneval-scala', # ok
                'humaneval-swift', # ok
                'humaneval-rb', # ok
                'humaneval-d', # ok
                'humaneval-jl', # ok
        ],
        }
        self._run_dataset_test('multiple_humaneval', dataset_args, limit=10, use_cache='outputs/20251212_102548', rerun_review=True, debug=False, eval_batch_size=2)

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

    def test_mbpp_plus(self):
        """Test MBPP+ dataset."""
        self._run_dataset_test('mbpp_plus', limit=10)

    def test_multipl_e_humaneval_volcengine_sandbox(self):
        sandbox_type='volcengine'
        use_sandbox=True
        sandbox_manager_config={
            'base_url': 'http://127.0.0.1:8080',
        }
        dataset_args = {
            'subset_list': [
                'humaneval-cpp', # ok
                # 'humaneval-ts', # ok
                # 'humaneval-sh', # ok
                # 'humaneval-cs', # need x86_64 docker image
                # 'humaneval-go', # ok
                # 'humaneval-java', # ok
                # 'humaneval-lua', # ok
                # 'humaneval-js', # ok
                # 'humaneval-php', # ok
                # 'humaneval-pl', # ok
                # 'humaneval-rkt', # ok
                # 'humaneval-r', # ok
                # 'humaneval-rs', # ok
                # 'humaneval-scala', # ok
                # 'humaneval-swift', # ok
                # 'humaneval-rb', # ok
                # 'humaneval-d', # ok
                # 'humaneval-jl', # ok
        ],
        }
        self._run_dataset_test('multiple_humaneval', dataset_args=dataset_args, limit=10, sandbox_type=sandbox_type, use_sandbox=use_sandbox, sandbox_manager_config=sandbox_manager_config)
