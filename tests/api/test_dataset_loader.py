import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import lance
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evalscope.api.dataset import FieldSpec, Sample
from evalscope.api.dataset.loader import DictDataLoader, LocalDataLoader


def _sample_from_record(record: dict) -> Sample:
    return Sample(input=record['text'])


def _make_direct_loader(
    loader_kind: str,
    tmp_path: Path,
    limit: Optional[float | int],
    repeats: Any = 1,
) -> LocalDataLoader | DictDataLoader:
    records = [{'text': f'row-{index}'} for index in range(4)]
    if loader_kind == 'local':
        local_path = tmp_path / 'records.lance'
        lance.write_dataset(pa.Table.from_pylist(records), local_path)
        return LocalDataLoader(
            data_id_or_path=str(local_path),
            split='train',
            sample_fields=_sample_from_record,
            limit=limit,
            repeats=repeats,
        )
    return DictDataLoader(dict_list=records, sample_fields=_sample_from_record, limit=limit, repeats=repeats)


@pytest.mark.parametrize('loader_kind', ['local', 'dict'])
@pytest.mark.parametrize('limit', [2.5, -1, -0.1, math.nan, math.inf])
def test_direct_loaders_reject_invalid_limits(
    loader_kind: str,
    limit: float | int,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match='Limit must'):
        _make_direct_loader(loader_kind, tmp_path, limit).load()


@pytest.mark.parametrize('loader_kind', ['local', 'dict'])
@pytest.mark.parametrize(
    'limit, expected',
    [
        (None, 4),
        (0, 4),
        (0.5, 2),
        (1.0, 4),
        (2, 2),
    ],
)
def test_direct_loaders_preserve_valid_limit_semantics(
    loader_kind: str,
    limit: Optional[float | int],
    expected: int,
    tmp_path: Path,
) -> None:
    loader = _make_direct_loader(loader_kind, tmp_path, limit)

    assert len(loader.load()) == expected
    assert len(loader.load()) == expected
    assert loader.limit == limit


@pytest.mark.parametrize('metadata_fields', [None, []])
def test_dict_loader_defaults_to_independent_empty_metadata(metadata_fields: Optional[List[str]]) -> None:
    records = [{'input': 'First question', 'target': 'A'}, {'input': 'Second question', 'target': 'B'}]
    loader = DictDataLoader(dict_list=records, sample_fields=FieldSpec(metadata=metadata_fields))

    dataset = loader.load()

    assert [(sample.input, sample.target, sample.metadata) for sample in dataset] == [
        ('First question', 'A', {}),
        ('Second question', 'B', {}),
    ]
    assert dataset[0].metadata is not dataset[1].metadata
    dataset[0].metadata['source'] = 'first'
    assert dataset[1].metadata == {}
    assert loader.load()[0].metadata == {}


def test_local_loader_defaults_to_empty_metadata(tmp_path: Path) -> None:
    record = {'input': 'Which city is the capital of France?', 'target': 'Paris'}
    path = tmp_path / 'questions.lance'
    lance.write_dataset(pa.Table.from_pylist([record]), path)

    dataset = LocalDataLoader(data_id_or_path=str(path), split='test').load()

    assert len(dataset) == 1
    assert dataset[0].input == record['input']
    assert dataset[0].target == record['target']
    assert dataset[0].metadata == {}


@pytest.mark.parametrize('as_json', [False, True])
@pytest.mark.parametrize('metadata', [{}, {'source': 'fixture', 'nested': {'rank': 0}}])
def test_dict_loader_preserves_explicit_metadata(metadata: Dict[str, Any], as_json: bool) -> None:
    record = {'input': 'Question', 'metadata': json.dumps(metadata) if as_json else metadata}

    dataset = DictDataLoader(dict_list=[record]).load()

    assert dataset[0].metadata == metadata


def test_dict_loader_preserves_selected_metadata_fields() -> None:
    record = {'input': 'Question', 'source': 'fixture', 'rank': 0, 'metadata': {'ignored': True}}

    dataset = DictDataLoader(
        dict_list=[record], sample_fields=FieldSpec(metadata=['source', 'rank', 'missing'])
    ).load()

    assert dataset[0].metadata == {'source': 'fixture', 'rank': 0, 'missing': None}


@pytest.mark.parametrize('metadata', [None, [], 42, 'null', '[]', '{invalid'])
def test_dict_loader_rejects_invalid_metadata(metadata: Any) -> None:
    with pytest.raises(ValueError):
        DictDataLoader(dict_list=[{'input': 'Question', 'metadata': metadata}]).load()


def test_repeats_group_the_samples_of_a_multi_sample_record_by_question() -> None:
    # A record may yield several samples (one per question); each repeat group must hold one question.
    def record_to_samples(record: Dict[str, Any]) -> List[Sample]:
        return [Sample(input=question, metadata={'question': question}) for question in record['questions']]

    dataset = DictDataLoader(dict_list=[{'questions': ['a', 'b']}], sample_fields=record_to_samples, repeats=2).load()

    groups: Dict[int, List[str]] = {}
    for sample in dataset:
        groups.setdefault(sample.group_id, []).append(sample.metadata['question'])
    assert groups == {0: ['a', 'a'], 1: ['b', 'b']}


@pytest.mark.parametrize('loader_kind', ['local', 'dict'])
@pytest.mark.parametrize('repeats', [0, -1, 1.5, True, '1'])
def test_direct_loaders_reject_invalid_repeats(
    loader_kind: str,
    repeats: Any,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match='Repeats must be a positive integer'):
        _make_direct_loader(loader_kind, tmp_path, limit=None, repeats=repeats)


@pytest.mark.parametrize('seed', [0, 42])
@pytest.mark.parametrize('limit', [None, 3, 0.5])
def test_dict_loader_seeded_shuffle_is_repeatable(seed: int, limit: Optional[float | int]) -> None:
    records = [{'input': f'question-{index}'} for index in range(12)]
    original_inputs = [record['input'] for record in records]
    loader = DictDataLoader(dict_list=records, shuffle=True, seed=seed, limit=limit)

    first = [sample.input for sample in loader.load()]
    second = [sample.input for sample in loader.load()]
    shared_input = [
        sample.input
        for sample in DictDataLoader(dict_list=records, shuffle=True, seed=seed, limit=limit).load()
    ]

    assert first != original_inputs[:len(first)]
    assert second == first
    assert shared_input == first


@pytest.mark.parametrize('seed', [None, 0])
def test_dict_loader_shuffle_preserves_input_records(seed: Optional[int]) -> None:
    records = [{'input': f'question-{index}'} for index in range(12)]
    original_records = list(records)

    dataset = DictDataLoader(dict_list=records, shuffle=True, seed=seed, limit=3).load()

    assert len(dataset) == 3
    assert records == original_records


@pytest.mark.parametrize('extension', ['csv', 'tsv', 'jsonl', 'parquet'])
def test_local_loader_rejects_non_lance_formats(tmp_path: Path, extension: str) -> None:
    path = tmp_path / f'default_test.{extension}'
    if extension == 'parquet':
        pq.write_table(pa.table({'input': ['old']}), path)
    else:
        path.touch()
    with pytest.raises(ValueError, match='Unsupported dataset format'):
        LocalDataLoader(data_id_or_path=str(path), split='test').load()
    with pytest.raises(FileNotFoundError, match='No Lance dataset'):
        LocalDataLoader(data_id_or_path=str(tmp_path), split='test').load()


@pytest.mark.parametrize('direct_path', [False, True])
@pytest.mark.parametrize('version, expected', [(None, 'new'), (1, 'old'), ('1', 'old'), ('baseline', 'old')])
def test_load_lance_snapshot(tmp_path: Path, direct_path: bool, version: int | str | None, expected: str) -> None:
    path = tmp_path / 'default_test.lance'
    pq.write_table(pa.table({'input': ['ignored']}), path.with_suffix('.parquet'))
    table = lance.write_dataset(pa.table({'input': ['old']}), path)
    table.tags.create('baseline', 1)
    lance.write_dataset(pa.table({'input': ['new']}), path, mode='overwrite')

    dataset = LocalDataLoader(
        data_id_or_path=str(path if direct_path else tmp_path), split='test', version=version
    ).load()
    assert [sample.input for sample in dataset] == [expected]
