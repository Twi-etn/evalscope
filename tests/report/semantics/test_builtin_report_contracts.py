from evalscope.api.metric import MetricKind, SampleScore, Score
from evalscope.api.registry import get_benchmark
from evalscope.report.generator import ReportGenerator


def test_mcp_atlas_generates_report_from_actual_aggregator() -> None:
    adapter = get_benchmark('mcp_atlas')
    sample = SampleScore(score=Score(value={'coverage_score': 1.0, 'pass': 1.0}), sample_id=0)
    scores = adapter.aggregate_scores([sample])
    report = ReportGenerator.generate_report({'default': scores}, 'model', adapter)

    assert report.primary_metric_identity is not None
    assert report.primary_metric_identity.name == 'pass_rate'
    assert report.primary_metric is not None
    assert report.primary_metric.semantics.semantic_id == 'quality.pass_at_k.ratio'
    assert report.primary_metric.semantics.kind is MetricKind.QUALITY
