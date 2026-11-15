"""Local-file benchmarks must honor the configured dataset shuffle seed."""

import random
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from evalscope.api.registry import get_benchmark
from evalscope.config import TaskConfig


@pytest.fixture(autouse=True)
def preserve_random_state() -> Iterator[None]:
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


@pytest.fixture
def qa_directory(tmp_path: Path) -> Path:
    rows = [
        {
            'question': f'question-{index}',
            'answer': f'answer-{index}',
        }
        for index in range(20)
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'default_test.parquet')
    return tmp_path


def load_targets(directory: Path, seed: int | None, shuffle: bool = True) -> list[str]:
    config = TaskConfig(
        datasets=['general_qa'],
        dataset_args={'general_qa': {'local_path': str(directory), 'shuffle': shuffle}},
        seed=seed,
        limit=5,
    )
    dataset, _ = get_benchmark('general_qa', config).load()
    return [str(sample.target) for sample in dataset['default']]


@pytest.mark.parametrize('seed', [0, 17, 42])
def test_local_file_shuffle_uses_configured_seed(qa_directory: Path, seed: int) -> None:
    expected = [f'answer-{index}' for index in range(20)]
    random.Random(seed).shuffle(expected)

    random.seed(123)
    first = load_targets(qa_directory, seed)
    random.seed(456)
    second = load_targets(qa_directory, seed)

    assert first == second == expected[:5]


@pytest.mark.parametrize('seed', [None, 42])
def test_local_file_without_shuffle_preserves_file_order(qa_directory: Path, seed: int | None) -> None:
    random.seed(123)
    first = load_targets(qa_directory, seed, shuffle=False)
    random.seed(456)
    second = load_targets(qa_directory, seed, shuffle=False)

    assert first == second == [f'answer-{index}' for index in range(5)]


def test_local_file_none_seed_preserves_global_random_behavior(qa_directory: Path) -> None:
    expected = [f'answer-{index}' for index in range(20)]
    random.Random(123).shuffle(expected)
    random.seed(123)

    assert load_targets(qa_directory, None) == expected[:5]


def test_native_loader_pins_and_records_lance_versions(tmp_path: Path) -> None:
    lance = pytest.importorskip('lance')
    from evalscope.evaluation_versioning import ResolvedBenchmarkSpec

    path = tmp_path / 'default_test.lance'
    lance.write_dataset(pa.table({'question': ['Q'], 'answer': ['first']}), path)
    config = TaskConfig(datasets=['general_qa'], dataset_args={'general_qa': {'local_path': str(tmp_path)}})
    adapter = get_benchmark('general_qa', config)

    assert ResolvedBenchmarkSpec.from_adapter(adapter, config).dataset_versions == {str(path): 1}
    lance.write_dataset(pa.table({'question': ['Q'], 'answer': ['second']}), path, mode='overwrite')

    assert adapter.load_dataset()['default'][0].target == 'first'
    latest = get_benchmark('general_qa', config)
    assert ResolvedBenchmarkSpec.from_adapter(latest, config).dataset_versions == {str(path): 2}
    assert latest.load_dataset()['default'][0].target == 'second'

    config.dataset_args['general_qa']['dataset_revision'] = 1
    pinned = get_benchmark('general_qa', config)
    assert ResolvedBenchmarkSpec.from_adapter(pinned, config).dataset_versions == {str(path): 1}
    assert pinned.load_dataset()['default'][0].target == 'first'

    declared_config = TaskConfig(datasets=['general_qa'])
    meta = replace(latest.benchmark_meta, dataset_id=str(path), dataset_revision=1, eval_split=None)
    declared = type(latest)(benchmark_meta=meta, task_config=declared_config)
    spec = ResolvedBenchmarkSpec.from_adapter(declared, declared_config)
    assert spec.dataset_revision == '1'
    assert spec.dataset_versions == {str(path): 1}
    assert declared.load_dataset()['default'][0].target == 'first'
