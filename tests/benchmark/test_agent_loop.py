# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end evaluation tests for AgentLoop pipeline.

All tests run against the real qwen3.8-flash API (DashScope).
No mocking – each test builds a TaskConfig and calls run_task().

Scenarios
---------
1. GSM8K baseline          – standard single-turn (no agent_config)
2. GSM8K agent, no tools   – function_calling strategy, pure multi-turn
3. GSM8K agent, python_exec + local env
4. GSM8K agent, bash + local env
5. GSM8K agent, python_exec + docker env
6. GSM8K + react strategy + python_exec + local env
7. GSM8K + swe_bench_toolcall strategy + bash + local env
"""

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

load_dotenv('.env')
env = dotenv_values('.env')

from evalscope.api.agent import NativeAgentConfig
from evalscope.api.agent.trace import AgentTrace, EventType
from evalscope.config import TaskConfig
from evalscope.constants import EvalType
from evalscope.run import run_task
from evalscope.utils.logger import get_logger

logger = get_logger()

_API_KEY = env.get('DASHSCOPE_API_KEY')
_API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

requires_api = unittest.skipUnless(_API_KEY, 'Requires DASHSCOPE_API_KEY in .env')


# ---------------------------------------------------------------------------
# Helpers for the external-agent (claude-code) e2e tests below
# ---------------------------------------------------------------------------

_IDEALAB_TOKEN = env.get('EVALSCOPE_IDEALAB_TOKEN') or os.environ.get('EVALSCOPE_IDEALAB_TOKEN')
_IDEALAB_URL = (
    env.get('EVALSCOPE_IDEALAB_BASE_URL') or os.environ.get('EVALSCOPE_IDEALAB_BASE_URL')
    or 'https://idealab.alibaba-inc.com/api/anthropic'
)
_IDEALAB_MODEL = env.get('EVALSCOPE_IDEALAB_MODEL') or os.environ.get('EVALSCOPE_IDEALAB_MODEL') or 'claude-opus-4-6'
_CLAUDE_CODE_IMAGE = (
    env.get('EVALSCOPE_CLAUDE_CODE_IMAGE') or os.environ.get('EVALSCOPE_CLAUDE_CODE_IMAGE')
    or 'evalscope/claude-code:dev'
)


def _docker_image_present(image: str) -> bool:
    """Cheap local probe for a docker image (no daemon call when CLI missing)."""
    if shutil.which('docker') is None:
        return False
    try:
        out = subprocess.run(
            ['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
            capture_output=True, text=True, timeout=15,
        )
        return out.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Two-pronged gate: real LLM credentials AND the pre-baked claude-code image
# must be available. Build the image with
# ``bash evalscope/agent/external/runners/_assets/build_claude_code_image.sh``.
requires_claude_code_e2e = unittest.skipUnless(
    bool(_IDEALAB_TOKEN) and _docker_image_present(_CLAUDE_CODE_IMAGE),
    f'Requires EVALSCOPE_IDEALAB_TOKEN in .env and docker image {_CLAUDE_CODE_IMAGE!r} '
    f'(build via build_claude_code_image.sh).',
)

def _base_cfg(**overrides) -> dict:
    """Common TaskConfig kwargs shared by all tests."""
    cfg = {
        'model': 'qwen3.8-flash',
        'api_url': _API_URL,
        'api_key': _API_KEY,
        'eval_type': EvalType.OPENAI_API,
        'eval_batch_size': 5,
        'limit': 5,
        'generation_config': {
            'max_tokens': 2048,
            'temperature': 0.7,
            'parallel_tool_calls': True,
        },
        'debug': True,
    }
    cfg.update(overrides)
    return cfg


def _read_review_results(benchmark: str, model: str = 'qwen3.8-flash', work_dir: str = './outputs') -> list[dict]:
    """Read all saved reviews for a benchmark."""
    from evalscope.api.evaluator.result_io import read_result_rows

    reviews_dir = Path(work_dir) / 'reviews' / model
    return [row for path in sorted(reviews_dir.glob(f'{benchmark}_*.parquet')) for row in read_result_rows(path)]


# ---------------------------------------------------------------------------
# 1. GSM8K baseline (no agent_config)
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KBaseline(unittest.TestCase):
    """Standard single-turn GSM8K – no agent loop, verifies baseline works."""

    def test_gsm8k_no_agent(self):
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

    def test_gsm8k_no_agent_trace_absent(self):
        """ReviewResult.agent_trace must be None for non-agent runs."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
            )
        )
        run_task(cfg)
        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            self.assertIsNone(r.get('agent_trace'), 'agent_trace should be absent for baseline')


# ---------------------------------------------------------------------------
# 2. GSM8K + agent, no tools (pure multi-turn chain-of-thought)
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentNoTools(unittest.TestCase):
    """function_calling strategy with no tools – model reasons step by step."""

    def test_run_succeeds_and_returns_score(self):
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=[],
                    max_steps=3,
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

# ---------------------------------------------------------------------------
# 3. GSM8K + agent, python_exec + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentPythonExecLocal(unittest.TestCase):
    """function_calling strategy with python_exec tool + local subprocess env."""

    def test_run_and_trace_with_python_exec(self):
        """Run succeeds; trace.environment == 'local'; TOOL_CALL paired with TOOL_RESULT."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=['python_exec'],
                    environment='local',
                    max_steps=5,
                    kwargs={'system_prompt': (
                        'Always call the python_exec tool to compute the answer.'
                    )},
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        any_tool_used = False
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.environment, 'local')
            types = [e.type for e in trace.events]
            if EventType.TOOL_CALL in types:
                any_tool_used = True
                self.assertIn(EventType.TOOL_RESULT, types)
        logger.info(f'Any tool used: {any_tool_used}')
        # Note: we don't assert any_tool_used – model may choose not to call tool
        # but we verify the trace is structurally correct either way


# ---------------------------------------------------------------------------
# 4. GSM8K + agent, bash + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentBashLocal(unittest.TestCase):
    """function_calling strategy with bash tool + local subprocess env."""

    def test_run_and_trace_with_bash(self):
        """Run succeeds; trace.environment == 'local' and events are present."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=['bash'],
                    environment='local',
                    max_steps=5,
                ),
                eval_batch_size=1,
                limit=1,
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        for r in reviews:
            trace = AgentTrace.model_validate(r['agent_trace'])
            self.assertEqual(trace.environment, 'local')
            self.assertGreater(len(trace.events), 0)


# ---------------------------------------------------------------------------
# 5. GSM8K + agent, python_exec + docker environment
# ---------------------------------------------------------------------------

@requires_api
class TestGSM8KAgentPythonExecDocker(unittest.TestCase):
    """function_calling strategy with python_exec tool + Docker sandbox env."""

    def test_run_and_trace_environment_is_docker(self):
        """Run succeeds, returns score, and AgentTrace.environment == 'docker'."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['gsm8k'],
                dataset_args={'gsm8k': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='function_calling',
                    tools=['python_exec'],
                    environment='docker',
                    environment_extra={
                        'image': 'python:3.11-slim',
                        'timeout': 60,
                    },
                    max_steps=5,
                    kwargs={'system_prompt': (
                        'You are a math solver. '
                        'Use the python_exec tool to verify your calculations.'
                    )},
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        reviews = _read_review_results('gsm8k', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.environment, 'docker')
            self.assertGreater(len(trace.events), 0)
            types = [e.type for e in trace.events]
            self.assertIn(EventType.MODEL_GENERATE, types)


# ---------------------------------------------------------------------------
# 6. GSM8K + react strategy + python_exec + local environment
# ---------------------------------------------------------------------------

@requires_api
class TestAIMEReActFC(unittest.TestCase):
    """ReAct strategy (FC mode) with bash tool + local env."""

    def test_react_run_and_trace(self):
        """Run succeeds; trace has strategy='react' and correct event structure."""
        cfg = TaskConfig(
            **_base_cfg(
                datasets=['aime26'],
                dataset_args={'aime26': {'few_shot_num': 0}},
                agent_config=NativeAgentConfig(
                    strategy='react',
                    tools=['bash'],
                    environment='local',
                    max_steps=10,
                ),
            )
        )
        result = run_task(cfg)
        self.assertIn('aime26', result)

        reviews = _read_review_results('aime26', work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict)
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.strategy, 'react')
            self.assertEqual(trace.environment, 'local')
            types = [e.type for e in trace.events]
            self.assertIn(EventType.MODEL_GENERATE, types)
            # ReAct FC mode should have TOOL_CALL events when tools are used.
            if EventType.TOOL_CALL in types:
                self.assertIn(EventType.TOOL_RESULT, types)


# ---------------------------------------------------------------------------
# 8. GSM8K + external agent (claude-code) inside the pre-baked docker image
# ---------------------------------------------------------------------------


class _ScrubAnthropicEnvMixin:
    """Strip ``ANTHROPIC_AUTH_TOKEN`` / ``ANTHROPIC_API_KEY`` /
    ``ANTHROPIC_BASE_URL`` for the duration of one test.

    Idealab rejects requests carrying both ``Authorization`` and
    ``x-api-key``; the Anthropic SDK auto-populates the former from
    ``ANTHROPIC_AUTH_TOKEN``, so any ``.env`` leakage breaks the test.
    """

    _SCRUB_KEYS = ('ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL')

    def setUp(self) -> None:
        super().setUp()
        self._saved_env = {k: os.environ.pop(k, None) for k in self._SCRUB_KEYS}

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()


@requires_claude_code_e2e
class TestGSM8KExternalClaudeCode(_ScrubAnthropicEnvMixin, unittest.TestCase):
    """End-to-end: gsm8k → external claude-code CLI inside the pre-baked
    ``evalscope/claude-code:dev`` container → bridge → idealab Anthropic.

    Validates the full external-agent loop for a generic single-turn
    benchmark (no per-instance sandbox image needed; the runner image is
    the playground). Skipped unless the image is built AND idealab
    credentials are in ``.env``.
    """

    def test_gsm8k_through_external_claude_code(self):
        cfg = TaskConfig(
            model=_IDEALAB_MODEL,
            api_url=_IDEALAB_URL,
            api_key=_IDEALAB_TOKEN,
            eval_type='anthropic_api',
            # Idealab User-Agent whitelist: only claude-cli traffic passes.
            model_args={
                'default_headers': {
                    'User-Agent': 'claude-cli/2.1.146 (external, cli)',
                },
            },
            datasets=['gsm8k'],
            dataset_args={'gsm8k': {'few_shot_num': 0}},
            agent_config={
                'mode': 'external',
                'framework': 'claude-code',
                'environment': 'docker',
                'environment_extra': {
                    'sandbox_config': {
                        'image': _CLAUDE_CODE_IMAGE,
                        'working_dir': '/workspace',
                    },
                    'timeout': 180.0,
                },
                # The pre-baked image already has node + claude on PATH;
                # ``auto_install=False`` short-circuits ``setup()``'s probe.
                'kwargs': {
                    'auto_install': False,
                    # Empty string = inherit the inside-container HOME.
                    'home_override': '',
                },
                'timeout': 300.0,
            },
            eval_batch_size=1,
            limit=1,
        )
        result = run_task(cfg)
        self.assertIn('gsm8k', result)

        # Trace must reflect the external-agent path (framework='claude-code',
        # environment='docker') — proves the external dispatch fired and the
        # docker env flowed through.
        reviews = _read_review_results('gsm8k', model=_IDEALAB_MODEL, work_dir=cfg.work_dir)
        self.assertGreater(len(reviews), 0)
        for r in reviews:
            trace_dict = r.get('agent_trace')
            self.assertIsNotNone(trace_dict, 'external-agent path must populate agent_trace')
            trace = AgentTrace.model_validate(trace_dict)
            self.assertEqual(trace.framework, 'claude-code')
            # EnclaveAgentEnvironment's instance ``name`` is always 'enclave'
            # regardless of the 'docker' / 'volcengine' registry alias used.
            self.assertEqual(trace.environment, 'enclave')
            types = [e.type for e in trace.events]
            self.assertIn(EventType.RUN_START, types)
            self.assertIn(EventType.MODEL_GENERATE, types)
            self.assertIn(EventType.RUN_END, types)


if __name__ == '__main__':
    unittest.main()
