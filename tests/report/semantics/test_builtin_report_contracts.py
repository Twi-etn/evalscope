from typing import Any, Dict, List

import pytest

import evalscope  # noqa: F401  # imported for benchmark registration side effects
from evalscope.api.metric import MetricKind, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.report import Report
from evalscope.report.generator import ReportGenerator


def _sample_score(value: Dict[str, Any], sample_id: int = 0, category: str = '') -> SampleScore:
    return SampleScore(
        score=Score(value=value),
        sample_id=sample_id,
        sample_metadata={'category': category} if category else {},
    )


def _generate_report(benchmark_name: str, sample_scores: List[SampleScore]) -> Report:
    adapter = get_benchmark(benchmark_name)
    scores = adapter.aggregate_scores(sample_scores)
    return ReportGenerator.generate_report({'default': scores}, 'model', adapter)


@pytest.mark.parametrize(
    ('benchmark_name', 'sample_scores', 'primary_name', 'expected_semantics'),
    [
        (
            'k2_verifier',
            [_sample_score({
                'finish_reason_tool_call': 1,
                'successful_tool_call': 1,
                'should_call_tool': 1,
            })],
            'trigger_similarity',
            'quality.similarity.ratio',
        ),
        (
            'minimax_verifier',
            [_sample_score({
                'inference_error': 0,
                'error_only_reasoning': 0,
                'tool_calls_run': 1,
                'tool_calls_finish_tool_calls': 1,
                'tool_calls_schema_valid': 1,
                'expected_tool_call_labeled': 1,
                'tool_calls_match': 1,
                'language_following_checked': 1,
                'language_following_valid': 1,
                'repeat_ngram_checked': 1,
                'repeat_ngram_valid': 1,
                'scenario_check_checked': 1,
                'scenario_check_valid': 1,
            })],
            'tool_calls_match_rate',
            'quality.accuracy.ratio',
        ),
        (
            'mcp_atlas',
            [_sample_score({
                'coverage_score': 1.0,
                'pass': 1.0,
            })],
            'pass_rate',
            'quality.pass_at_k.ratio',
        ),
    ],
)
def test_vendor_benchmarks_generate_reports_from_actual_aggregators(
    benchmark_name: str,
    sample_scores: List[SampleScore],
    primary_name: str,
    expected_semantics: str,
) -> None:
    report = _generate_report(benchmark_name, sample_scores)

    assert report.primary_metric_identity is not None
    assert report.primary_metric_identity.name == primary_name
    assert report.primary_metric is not None
    assert report.primary_metric.semantics.semantic_id == expected_semantics
    assert report.primary_metric.semantics.kind is MetricKind.QUALITY


def test_minimax_auxiliary_rates_have_declared_directions() -> None:
    report = _generate_report(
        'minimax_verifier',
        [_sample_score({
            'inference_error': 0,
            'error_only_reasoning': 1,
            'tool_calls_run': 1,
            'tool_calls_finish_tool_calls': 1,
            'tool_calls_schema_valid': 1,
            'expected_tool_call_labeled': 1,
            'tool_calls_match': 1,
            'language_following_checked': 1,
            'language_following_valid': 1,
            'repeat_ngram_checked': 1,
            'repeat_ngram_valid': 1,
            'scenario_check_checked': 1,
            'scenario_check_valid': 1,
        })],
    )
    semantics = {metric.identity.name: metric.semantics for metric in report.metrics}

    assert semantics['error_only_reasoning_rate'].semantic_id == 'quality.error_rate.ratio'
    for name in ('language_following_success_rate', 'repeat_ngram_pass_rate', 'scenario_check_pass_rate'):
        assert semantics[name].semantic_id == 'quality.accuracy.ratio'
        assert semantics[name].kind is MetricKind.QUALITY
