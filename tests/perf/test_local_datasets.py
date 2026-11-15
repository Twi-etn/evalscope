import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.datasets.longalpaca import LongAlpacaDatasetPlugin
from evalscope.perf.plugin.datasets.openqa import OpenqaDatasetPlugin


@pytest.mark.parametrize('direct', [False, True])
def test_openqa_reads_local_jsonl(tmp_path: Path, direct: bool) -> None:
    file = tmp_path / 'open_qa.jsonl'
    file.write_text(json.dumps({'question': 'A local question'}) + '\n', encoding='utf-8')
    path = str(file if direct else tmp_path)
    plugin = OpenqaDatasetPlugin(Arguments(model='local', dataset_path=path))

    assert 'A local question' in str(next(plugin.build_messages()))
    assert plugin.query_parameters.dataset_path == path


def test_perf_reads_named_local_parquet(tmp_path: Path) -> None:
    pq.write_table(pa.table({'instruction': ['A local row']}), tmp_path / 'default_train.parquet')
    plugin = LongAlpacaDatasetPlugin(Arguments(model='local', dataset_path=str(tmp_path)))

    assert 'A local row' in str(next(plugin.build_messages()))


def test_perf_rejects_missing_local_dataset_path() -> None:
    plugin = OpenqaDatasetPlugin(Arguments(model='local', data_source='huggingface'))
    with pytest.raises(ValueError, match='--dataset-path'):
        next(plugin.build_messages())
