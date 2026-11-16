"""End-to-end evaluator test using the local ``general_mcq`` benchmark.

Validates that the external-agent path slots into the full evaluator
pipeline (TaskConfig → benchmark adapter → bridge → mock agent → score)
without needing network access or a real LLM.

The mock agent forwards the model's output verbatim, so we patch
``MockLLM`` to always emit ``B`` — the correct answer for the first MCQ
sample in the local fixture — and assert the resulting score is 1.0.
That pins both the score path and the agent → model wiring.
"""

from pathlib import Path

import lance
import pyarrow as pa
import pytest

from evalscope.api.evaluator.result_io import read_result_rows
from evalscope.api.model import ModelOutput
from evalscope.constants import EvalType
from evalscope.models.mockllm import MockLLM
from evalscope.report.report import Report
from evalscope.run import run_task

# The MockLLM emits exactly this text; the first general_mcq sample expects ``B``.
_MOCK_ANSWER = 'B'


@pytest.fixture(autouse=True)
def _patch_mock_llm(monkeypatch):
    """Force MockLLM to emit the answer the local fixture expects."""
    original_init = MockLLM.__init__

    def _patched_init(self, *args, **kwargs):
        outs = [ModelOutput.from_content(model='mockllm', content=_MOCK_ANSWER) for _ in range(64)]
        kwargs['custom_outputs'] = outs
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(MockLLM, '__init__', _patched_init)
    yield


def test_external_agent_through_evaluator(tmp_path: Path) -> None:
    """``general_mcq`` + mock LLM + mock external agent, scored end-to-end."""
    from evalscope.config import TaskConfig

    dataset_path = tmp_path / 'example_val.lance'
    lance.write_dataset(
        pa.Table.from_pylist([{'question': 'Choose the second option.', 'A': 'first', 'B': 'second', 'answer': 'B'}]),
        dataset_path,
    )
    task_cfg = TaskConfig(
        model='mock_llm',
        eval_type=EvalType.MOCK_LLM,
        datasets=['general_mcq'],
        dataset_args={
            'general_mcq': {
                'local_path': str(dataset_path),
                'subset_list': ['example'],
            },
        },
        agent_config={
            'mode': 'external',
            'framework': 'mock',
            'environment': 'local'
        },
        eval_batch_size=1,
        limit=1,
        analysis_report=False,
        work_dir=str(tmp_path),
    )
    result = run_task(task_cfg=task_cfg)

    assert isinstance(result, dict) and result, f'expected a non-empty per-benchmark report dict, got {result!r}'
    assert 'general_mcq' in result, f'general_mcq missing from report dict; keys={list(result.keys())}'
    report = result['general_mcq']
    assert isinstance(report, Report), f'expected Report, got {type(report).__name__}'
    # Mock agent answered 'B' and the first sample's gold answer is also 'B',
    # so 1/1 should score a perfect 1.0 — proves the agent's output flowed
    # all the way back through the scoring path.
    assert report.primary_metric is not None
    assert report.primary_metric.score == pytest.approx(1.0)

    # Spot-check that the external-agent path actually ran by looking at the
    # serialized review file; the adapter's trace is persisted with the score.
    review_path, = tmp_path.glob('*/reviews/*/general_mcq_example.parquet')
    review, = read_result_rows(review_path)
    assert review['agent_trace']['framework'] == 'mock'
