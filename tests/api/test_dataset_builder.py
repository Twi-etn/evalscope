from typing import Any, Dict

import pytest

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import (
    MemoryDataset,
    Sample,
    build_dataset_dict_from_record_map,
    build_dataset_from_records,
    load_local_file_dataset,
    resolve_local_dataset_path,
)
from evalscope.config import TaskConfig


def record_to_sample(record: Dict[str, Any]) -> Sample:
    return Sample(input=str(record['value']), metadata={'value': record['value']})


def test_build_dataset_from_records_applies_limit_repeats_and_reindex() -> None:
    dataset = build_dataset_from_records(
        records=[{'value': 1}, {'value': 2}, {'value': 3}],
        sample_fields=record_to_sample,
        name='subset',
        location='source',
        limit=2,
        repeats=2,
        shuffle=False,
        seed=None,
    )

    assert isinstance(dataset, MemoryDataset)
    assert [sample.input for sample in dataset] == ['1', '1', '2', '2']
    assert [sample.id for sample in dataset] == [0, 1, 2, 3]
    assert [sample.group_id for sample in dataset] == [0, 0, 1, 1]
    assert dataset.name == 'subset'
    assert dataset.location == 'source'


@pytest.mark.parametrize('repeats', [0, -1, 1.5, True, '1'])
def test_build_dataset_from_records_rejects_invalid_repeats(repeats: object) -> None:
    with pytest.raises(ValueError, match='Repeats must be a positive integer'):
        build_dataset_from_records(
            records=[{'value': 1}],
            sample_fields=record_to_sample,
            name='subset',
            location='source',
            limit=None,
            repeats=repeats,
            shuffle=False,
            seed=None,
        )


def test_build_dataset_from_records_filter_runs_before_reindex() -> None:
    dataset = build_dataset_from_records(
        records=[{'value': 1}, {'value': 2}, {'value': 3}],
        sample_fields=record_to_sample,
        name='subset',
        location=None,
        limit=None,
        repeats=1,
        shuffle=False,
        seed=None,
        filter_func=lambda sample: sample.metadata['value'] != 2,
    )

    assert [sample.input for sample in dataset] == ['1', '3']
    assert [sample.id for sample in dataset] == [0, 1]


def test_build_dataset_dict_from_record_map_uses_stable_shuffle_seed() -> None:
    record_map = {'a': [{'value': idx} for idx in range(6)]}

    first = build_dataset_dict_from_record_map(
        record_map,
        sample_fields=record_to_sample,
        location='source',
        limit=3,
        repeats=1,
        shuffle=True,
        seed=7,
    )
    second = build_dataset_dict_from_record_map(
        record_map,
        sample_fields=record_to_sample,
        location='source',
        limit=3,
        repeats=1,
        shuffle=True,
        seed=7,
    )

    assert [sample.input for sample in first['a']] == [sample.input for sample in second['a']]
    assert [sample.input for sample in first['a']] != ['0', '1', '2']


class DummyAdapter(DefaultDataAdapter):

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        return record_to_sample(record)


def make_adapter(dataset_id: str) -> DummyAdapter:
    return DummyAdapter(
        benchmark_meta=BenchmarkMeta(name='dummy', dataset_id=dataset_id, eval_split='test'),
        task_config=TaskConfig(datasets=['dummy']),
    )


@pytest.mark.parametrize('allow_local_file', [False, True])
def test_resolve_local_dataset_directory(tmp_path, allow_local_file: bool) -> None:
    dataset_dir = tmp_path / 'dataset'
    dataset_dir.mkdir()
    adapter = make_adapter(str(dataset_dir))

    assert resolve_local_dataset_path(adapter, allow_local_file=allow_local_file) == str(dataset_dir.resolve())


def test_resolve_local_file_requires_explicit_opt_in(tmp_path) -> None:
    path = tmp_path / 'collection.jsonl'
    path.write_text('{"value": 1}\n', encoding='utf-8')
    adapter = make_adapter(str(path))

    with pytest.raises(FileNotFoundError, match='Local dataset directory was not found'):
        resolve_local_dataset_path(adapter)
    assert resolve_local_dataset_path(adapter, allow_local_file=True) == str(path.resolve())


def test_resolve_missing_local_path_with_file_opt_in(tmp_path) -> None:
    adapter = make_adapter(str(tmp_path / 'missing.jsonl'))
    with pytest.raises(FileNotFoundError, match='missing.jsonl'):
        resolve_local_dataset_path(adapter, allow_local_file=True)




def test_load_local_file_dataset_uses_explicit_reader(tmp_path) -> None:
    from evalscope.utils.io_utils import jsonl_to_list

    path = tmp_path / 'file.jsonl'
    path.write_text('{"value": 1}\n', encoding='utf-8')
    dataset = load_local_file_dataset(
        adapter=make_adapter(str(path)),
        dataset_path=str(path),
        reader=jsonl_to_list,
        sample_fields=record_to_sample,
        limit=5,
        repeats=2,
        shuffle=True,
    )
    assert [sample.input for sample in dataset] == ['1', '1']
    assert dataset.name == 'file'
    assert dataset.location == str(path)
