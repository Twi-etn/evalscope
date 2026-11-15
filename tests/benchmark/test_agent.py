# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values, load_dotenv

load_dotenv('.env')

env = dotenv_values('.env')

import unittest

import numpy as np

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.mcp import MCPServerConfigStdio
from evalscope.api.benchmark.adapters.browsergym_adapter import BrowserGymStep
from evalscope.api.evaluator.result_io import read_result_rows
from evalscope.api.messages import ChatMessageAssistant
from evalscope.api.model import ModelOutput
from evalscope.api.model.model_output import ChatCompletionChoice
from evalscope.api.tool import ToolCall, ToolFunction
from evalscope.benchmarks.toolathlon.toolathlon_adapter import ToolathlonAdapter
from evalscope.config import SandboxTaskConfig, TaskConfig
from evalscope.constants import EvalType, JudgeStrategy
from evalscope.models.mockllm import MockLLM
from evalscope.report.data_frames import get_model_prediction
from evalscope.run import run_task
from evalscope.utils.logger import get_logger
from tests.common import TestBenchmark

logger = get_logger()


class TestAgentBenchmark(TestBenchmark):
    """Agentic benchmark evaluation test cases."""

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
                'temperature': 0.7,
                'parallel_tool_calls': True,
                'retries': 3,
                'extra_body': {
                    'enable_thinking': True
                },
                'stream': True
            },
            'judge': {'strategy': JudgeStrategy.AUTO, 'models': {
                'model_id': 'qwen3.8-flash',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'extra_body': {
                        'enable_thinking': False
                    }
                }
            }},
            'debug': True,
        }

    def test_browsecomp(self):
        """Test BrowseComp benchmark end-to-end."""
        config_overrides = {
            'collect_perf': False,
            'debug': False,
            'eval_batch_size': 1,
            'limit': 1,
            'no_timestamp': True,
            'work_dir': 'outputs/test_agent_browsecomp',
        }
        if not env.get('DASHSCOPE_API_KEY'):
            config_overrides['judge'] = {'strategy': JudgeStrategy.RULE}

        self._run_dataset_test('browsecomp', **config_overrides)

        review_files = list(Path('outputs/test_agent_browsecomp').glob('reviews/*/browsecomp_default.parquet'))
        self.assertEqual(len(review_files), 1)
        review = read_result_rows(review_files[0])[0]
        self.assertNotIn('canary', review['sample_score']['sample_metadata'])

    def test_miniwob(self):
        """Run the BrowserGym agent, reward and reporting flow end to end."""

        class FakeSession:

            async def reset(self):
                return BrowserGymStep(
                    observation={
                        'goal': 'Click OK',
                        'url': 'http://miniwob/click-dialog.html',
                        'axtree_txt': '[1] button "OK"',
                        'last_action_error': False,
                        'screenshot': np.zeros((8, 8, 3), dtype=np.uint8),
                    },
                )

            async def step(self, action):
                assert action == 'click("1")'
                return BrowserGymStep(
                    observation={
                        'goal': 'Click OK',
                        'url': 'http://miniwob/click-dialog.html',
                        'axtree_txt': '[1] button "OK"',
                        'last_action_error': False,
                        'screenshot': np.zeros((8, 8, 3), dtype=np.uint8),
                    },
                    reward=1.0,
                    done=True,
                )

            async def close(self):
                return None

        record = {
            'task_name': 'miniwob.click-dialog',
            'miniwob_category': 'test',
            'comment': '',
            'webgum_subset': 'False',
            'similarity_group': '0',
            'browsergym_split': 'test',
            'task_id': 'miniwob.click-dialog',
            '_episode_seeds': [28, 29, 30, 31, 32],
        }
        call = ToolCall(
            id='browser-1',
            function=ToolFunction(name='browser_action', arguments={'action': 'click("1")'}),
        )
        output = ModelOutput(
            model='mock_llm',
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content='', tool_calls=[call]),
                    stop_reason='tool_calls',
                )
            ],
        )
        original_init = MockLLM.__init__

        def patched_init(model_self, *args, **kwargs):
            kwargs['custom_outputs'] = [output.model_copy(deep=True) for _ in range(5)]
            original_init(model_self, *args, **kwargs)

        with tempfile.TemporaryDirectory() as work_dir, \
                patch('evalscope.api.benchmark.adapters.browsergym_adapter.check_import'), \
                patch('evalscope.benchmarks.miniwob.miniwob_adapter.load_miniwob_records',
                      return_value=([record], Path('/cache/miniwob.csv'))), \
                patch('evalscope.benchmarks.miniwob.miniwob_adapter.MiniWobAdapter.create_browsergym_session',
                      side_effect=lambda sample: FakeSession()), \
                patch.object(MockLLM, '__init__', patched_init):
            reports = run_task(
                TaskConfig(
                    model='mock_llm',
                    datasets=['miniwob'],
                    eval_type='mock_llm',
                    repeats=5,
                    eval_batch_size=1,
                    work_dir=work_dir,
                    no_timestamp=True,
                    analysis_report=False,
                    agent_config=NativeAgentConfig(
                        strategy='function_calling',
                        max_steps=10,
                    ),
                )
            )

            self.assertEqual(reports['miniwob'].score, 1.0)
            reviews = list(Path(work_dir).glob('reviews/**/*.parquet'))
            rows = read_result_rows(reviews[0])
            self.assertEqual(len(rows), 5)
            self.assertEqual([row['sample_score']['sample_metadata']['seed'] for row in rows], [28, 29, 30, 31, 32])
            self.assertTrue(all(row['agent_trace']['events'][0]['payload']['backend'] == 'browsergym' for row in rows))

            prediction_rows = get_model_prediction(work_dir, 'mock_llm', 'miniwob', 'default')
            browser_trace = prediction_rows.iloc[0]['AgentTrace']
            self.assertEqual(browser_trace['environment'], 'browsergym')
            self.assertEqual(browser_trace['events'][0]['type'], 'env_reset')
            self.assertIn(
                browser_trace['events'][0]['message_id'],
                {message['id']
                 for message in prediction_rows.iloc[0]['Messages']}
            )
            submit_event = next(event for event in browser_trace['events'] if event['type'] == 'submit')
            terminal_observation = next(
                message for message in prediction_rows.iloc[0]['Messages']
                if message['id'] == submit_event['message_id']
            )
            self.assertEqual(terminal_observation['tool_call_id'], ['browser-1'])
            self.assertTrue(terminal_observation['metadata']['done'])

    def test_researchrubrics(self):
        """Test ResearchRubrics with a real agent API and binary LLM judge."""
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is required for the ResearchRubrics real-API smoke test.')

        self._run_dataset_test(
            'researchrubrics',
            limit=5,
            eval_batch_size=5,
            collect_perf=False,
            debug=False,
        )

    def test_job_bench(self):
        """Test JobBench end-to-end with Docker artifacts and LLM judging."""
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is required for the JobBench end-to-end test.')

        self._run_dataset_test(
            'job_bench',
            limit=5,
            eval_batch_size=5,
            collect_perf=False,
            debug=False,
            agent_config=NativeAgentConfig(environment='docker', max_steps=80),
            sandbox=SandboxTaskConfig(default_config={
                'image': 'python:3.11-slim-bookworm',
                'network_enabled': True,
            }),
        )

    def test_wide_search(self):
        """Test WideSearch with real qwen3.8-flash, bash, Fetch MCP, and LLM judging."""
        if not env.get('DASHSCOPE_API_KEY'):
            self.skipTest('DASHSCOPE_API_KEY is required for the WideSearch real-API smoke test.')

        self._run_dataset_test(
            'wide_search',
            model='qwen3.8-flash',
            limit=1,
            eval_batch_size=1,
            collect_perf=False,
            debug=False,
            judge={'models': {
                'model_id': 'qwen3.8-flash',
                'api_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                'api_key': env.get('DASHSCOPE_API_KEY'),
                'generation_config': {
                    'temperature': 0.0,
                    'extra_body': {
                        'enable_thinking': False
                    }
                }
            }},
            agent_config=NativeAgentConfig(
                mcp_servers=[
                    MCPServerConfigStdio(
                        command=sys.executable,
                        args=['-m', 'mcp_server_fetch', '--ignore-robots-txt'],
                        name='fetch',
                    )
                ],
            ),
        )

    def test_terminal_bench_v2_1(self):
        """Test Terminal-Bench v2.1 dataset."""
        dataset_args = {
            'extra_params': {
                'timeout_multiplier': 3,
                'environment_kwargs': {
                    'override_cpus': 2
                },
            },
        }
        self._run_dataset_test('terminal_bench_v2_1', dataset_args, limit=3, eval_batch_size=3)

    def test_toolathlon(self):
        """Test Toolathlon official-service wrapper with a mocked service client."""

        class FakeToolathlonClient:
            last_config = None

            def __init__(self, config):
                FakeToolathlonClient.last_config = config

            def run_private(self):
                return {
                    'job_id': self.last_config.job_id or 'fake-toolathlon-job',
                    'output_dir': str(self.last_config.output_dir),
                    'acc': 1.0,
                    'eval_stats': {
                        'passed': 1,
                        'total': 1
                    },
                    'task_results': [{
                        'task': 'find-alita-paper',
                        'pass': True
                    }],
                }

        dataset_args = {
            'extra_params': {
                'task_list': ['find-alita-paper'],
                'workers': 1,
                'skip_container_restart': True,
                'model_params': {
                    'temperature': 0.0
                },
            }
        }

        with patch.object(ToolathlonAdapter, 'client_cls', FakeToolathlonClient):
            self._run_dataset_test(
                'toolathlon',
                dataset_args,
                use_mock=True,
                limit=1,
                eval_batch_size=1,
                no_timestamp=True,
                work_dir='outputs/test_agent_toolathlon',
                api_url='http://localhost:8000/v1',
                api_key='local-key',
            )

        self.assertIsNotNone(FakeToolathlonClient.last_config)
        self.assertEqual(FakeToolathlonClient.last_config.base_url, 'http://localhost:8000/v1')
        self.assertEqual(FakeToolathlonClient.last_config.api_key, 'local-key')
        self.assertEqual(FakeToolathlonClient.last_config.task_list, ['find-alita-paper'])
        self.assertEqual(FakeToolathlonClient.last_config.model_params['temperature'], 0.0)
        self.assertNotIn('retries', FakeToolathlonClient.last_config.model_params)
        self.assertNotIn('batch_size', FakeToolathlonClient.last_config.model_params)

if __name__ == '__main__':
    # Run specific test: python -m unittest test_agent.TestAgentBenchmark.test_swe_bench_verified_agentic
    # Run all tests: python -m unittest test_agent.TestAgentBenchmark
    unittest.main()
