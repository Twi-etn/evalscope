import copy
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Union

from .dataset import (
    Dataset,
    FieldSpec,
    MemoryDataset,
    resolve_dataset_limit,
    validate_dataset_limit,
    validate_dataset_repeats,
)
from .utils import data_to_samples, record_to_sample_fn, shuffle_choices_if_requested

if TYPE_CHECKING:
    from lance import LanceDataset


def repeat_samples(samples: list, repeats: int) -> list:
    """Repeat each converted sample, so one record that yields several samples keeps each repeat group to one question."""
    return [copy.deepcopy(sample) for sample in samples for _ in range(repeats)] if repeats > 1 else samples


def _shuffle_in_place(data: list, seed: Optional[int]) -> None:
    """Shuffle a list in place with an optional seed.

    Uses random.Random(seed) instead of the deprecated random.shuffle(x, random)
    which was removed in Python 3.12.
    """
    if seed is not None:
        random.Random(seed).shuffle(data)
    else:
        random.shuffle(data)


def resolve_local_dataset_split(path: str, split: str, subset: str) -> Path:
    """Find the local Lance directory for a split."""
    path = Path(path).expanduser()
    if path.is_file():
        raise ValueError(f'Unsupported dataset format for {path}. Use a Lance dataset directory.')

    if not path.is_dir():
        raise FileNotFoundError(f'Local dataset path does not exist: {path}')

    if path.suffix == '.lance' or (path / '_versions').is_dir():
        return path
    for stem in (f'{subset}_{split}', subset):
        lance_path = path / f'{stem}.lance'
        if lance_path.is_dir():
            return lance_path

    raise FileNotFoundError(
        f'No Lance dataset in {path} for subset {subset!r}, split {split!r}. '
        f'Expected {subset}_{split}.lance or {subset}.lance.'
    )


def open_local_lance(path: Path, version: Optional[Union[int, str]]) -> 'LanceDataset':
    """Open a local Lance dataset at the requested version."""
    try:
        import lance
    except ModuleNotFoundError as exc:
        if exc.name != 'lance':
            raise
        raise ImportError('Lance datasets require pylance. Install it with `pip install pylance`.') from exc

    if isinstance(version, str) and version.isdecimal():
        version = int(version)
    # Resolve as an absolute filesystem path before passing it to Lance's URI-aware API.
    return lance.dataset(str(path.resolve()), version=version)


def read_local_dataset_records(
    path: str, split: str, subset: str = 'default', version: Optional[Union[int, str]] = None
) -> list[dict]:
    """Read a local Lance snapshot, using the latest version by default."""
    source = resolve_local_dataset_split(path, split, subset)
    return open_local_lance(source, version).to_table().to_pylist()


class DataLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(
        self,
        data_id_or_path: str,
        split: str,
        sample_fields: Optional[Union[FieldSpec, Callable]] = None,
        filter_func: Optional[Callable] = None,
        subset: str = 'default',
        version: Optional[Union[int, str]] = None,
        limit: Optional[Union[int, float]] = None,
        shuffle: bool = False,
        shuffle_choices: Optional[Union[bool, int]] = None,
        seed: Optional[int] = None,
        auto_id: bool = True,
        repeats: int = 1,
    ) -> None:
        self.data_id_or_path = data_id_or_path
        self.split = split
        self.sample_fields = sample_fields
        self.filter_func = filter_func
        self.subset = subset
        self.version = version
        self.limit = validate_dataset_limit(limit)
        self.shuffle = shuffle
        self.shuffle_choices = shuffle_choices
        self.seed = seed
        self.auto_id = auto_id
        self.repeats = validate_dataset_repeats(repeats)

    @abstractmethod
    def load(self) -> Dataset:
        """Load data from the source."""
        ...

    def _load_records(self, records: list[dict], location: Optional[str] = None) -> MemoryDataset:
        if self.shuffle:
            _shuffle_in_place(records, self.seed)
        if self.limit:
            records = records[: resolve_dataset_limit(self.limit, len(records))]

        samples = data_to_samples(data=records, data_to_sample=record_to_sample_fn(self.sample_fields))
        dataset = MemoryDataset(
            samples=repeat_samples(samples, self.repeats),
            name=Path(location).stem if location is not None else None,
            location=location,
        )
        if self.filter_func is not None:
            dataset = dataset.filter(self.filter_func)
        if self.auto_id:
            dataset.reindex(group_size=self.repeats)
        shuffle_choices_if_requested(dataset, self.shuffle_choices, self.seed)
        return dataset


class LocalDataLoader(DataLoader):
    """Load a local Lance dataset."""

    def load(self) -> MemoryDataset:
        """Load and convert local records to samples."""
        records = read_local_dataset_records(self.data_id_or_path, self.split, self.subset, self.version)
        return self._load_records(records, location=self.data_id_or_path)


class DictDataLoader(DataLoader):
    """Load a dataset from a list of dictionaries."""

    def __init__(self, dict_list: list[dict], **kwargs) -> None:
        super().__init__(data_id_or_path='', split='', **kwargs)
        self.dict_list = dict_list

    def load(self, *, location: Optional[str] = None) -> MemoryDataset:
        """Convert in-memory records to samples and optionally attach their source path."""
        return self._load_records(list(self.dict_list), location=location)
