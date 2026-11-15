import copy
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Optional, Union

from .dataset import DatasetDict, FieldSpec, MemoryDataset, Sample, validate_dataset_repeats
from .loader import DictDataLoader, _shuffle_in_place
from .utils import data_to_samples, record_to_sample_fn

if TYPE_CHECKING:
    from evalscope.api.benchmark import DataAdapter


def build_dataset_from_records(
    records: Iterable[dict],
    sample_fields: Union[FieldSpec, Callable],
    *,
    name: str,
    location: Optional[str],
    limit: Optional[Union[int, float]],
    repeats: int,
    shuffle: bool,
    seed: Optional[int],
    filter_func: Optional[Callable[[Sample], bool]] = None,
    auto_id: bool = True,
) -> MemoryDataset:
    """Build a MemoryDataset from raw records using the standard adapter mechanics.

    Note:
        ``repeats`` duplicates each *resulting sample* (not the raw record) ``repeats``
        times consecutively, then ``reindex`` groups them with ``group_size=repeats``.
        This matches the single-sample-per-record adapters. If ``sample_fields`` maps one
        record to multiple samples, the per-sample duplication order differs from record
        level duplication, which callers relying on k-metric grouping should be aware of.
    """
    repeats = validate_dataset_repeats(repeats)
    record_list = list(records)
    if shuffle:
        _shuffle_in_place(record_list, seed)

    if limit is not None:
        if isinstance(limit, float):
            if not (0.0 <= limit <= 1.0):
                raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
            limit = int(len(record_list) * limit)
        elif isinstance(limit, int) and limit < 0:
            raise ValueError('Limit must be a non-negative integer or a float between 0 and 1.')
        record_list = record_list[:limit]

    data_to_sample = record_to_sample_fn(sample_fields)
    samples = data_to_samples(data=record_list, data_to_sample=data_to_sample)
    if repeats > 1:
        samples = [copy.deepcopy(sample) for sample in samples for _ in range(repeats)]

    dataset = MemoryDataset(samples=samples, name=name, location=location, shuffled=shuffle)
    if filter_func is not None:
        dataset = dataset.filter(filter_func)
    if auto_id:
        dataset.reindex(group_size=repeats)
    return dataset


def build_dataset_dict_from_record_map(
    record_map: Dict[str, Iterable[dict]],
    sample_fields: Union[FieldSpec, Callable],
    *,
    location: Optional[str],
    limit: Optional[Union[int, float]],
    repeats: int,
    shuffle: bool,
    seed: Optional[int],
    filter_func: Optional[Callable[[Sample], bool]] = None,
    auto_id: bool = True,
) -> DatasetDict:
    """Build a DatasetDict from a mapping of subset name to raw records."""
    datasets = {}
    for subset, records in record_map.items():
        datasets[subset] = build_dataset_from_records(
            records=records,
            sample_fields=sample_fields,
            name=subset,
            location=location,
            limit=limit,
            repeats=repeats,
            shuffle=shuffle,
            seed=seed,
            filter_func=filter_func,
            auto_id=auto_id,
        )
    return DatasetDict(datasets)


def resolve_local_dataset_path(adapter: 'DataAdapter', *, allow_local_file: bool = False) -> str:
    """Resolve a local dataset root or a file for an adapter with its own file format."""
    path = Path(adapter.dataset_id).expanduser()
    if allow_local_file and path.is_file():
        # Preserve symlink extensions and the filesystem meaning of ".." after directory symlinks.
        return str(path.absolute())
    if path.is_dir():
        return str(path.resolve())
    raise FileNotFoundError(
        f'Local dataset directory was not found: {path}. Set dataset_args.{adapter.name}.local_path to a local dataset.'
    )


def resolve_local_dataset_file(dataset_root: Union[str, Path], file_name: str) -> str:
    """Find a required file within a local dataset directory."""
    root = Path(dataset_root).expanduser().resolve()
    path = (root / file_name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Invalid dataset file path: {file_name}')
    if not path.is_file():
        raise FileNotFoundError(f'Dataset file {file_name} was not found in {root}.')
    return str(path)


def load_local_file_dataset(
    adapter: 'DataAdapter',
    dataset_path: str,
    reader: Callable[[str], list[dict]],
    sample_fields: Union[FieldSpec, Callable],
    limit: Optional[Union[int, float]],
    repeats: int,
    shuffle: bool,
) -> MemoryDataset:
    """Load a benchmark-specific file using its explicit reader and the standard sample conversion."""
    return DictDataLoader(
        dict_list=reader(dataset_path),
        sample_fields=sample_fields,
        limit=limit,
        repeats=repeats,
        shuffle=shuffle,
        seed=adapter.seed,
    ).load(location=dataset_path)
