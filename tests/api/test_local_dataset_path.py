from pathlib import Path

import pytest

from evalscope.api.dataset import resolve_local_dataset_file
from evalscope.constants import HubType


def test_resolve_dataset_file_checks_the_local_root(tmp_path: Path) -> None:
    root = tmp_path / 'dataset'
    root.mkdir()
    nested = root / 'subset'
    nested.mkdir()
    file = nested / 'data.json'
    file.touch()
    (tmp_path / 'outside.json').touch()
    (root / 'link.json').symlink_to(tmp_path / 'outside.json')

    assert resolve_local_dataset_file(root, 'subset/data.json') == str(file.resolve())
    with pytest.raises(FileNotFoundError, match='missing.json'):
        resolve_local_dataset_file(root, 'missing.json')
    for path in ('../outside.json', 'link.json'):
        with pytest.raises(ValueError, match='Invalid dataset file path'):
            resolve_local_dataset_file(root, path)


def test_native_eval_rejects_remote_dataset_id_even_with_legacy_hub_config() -> None:
    from evalscope.api.registry import get_benchmark
    from evalscope.config import TaskConfig

    adapter = get_benchmark('gsm8k', TaskConfig(datasets=['gsm8k'], dataset_hub=HubType.HUGGINGFACE))

    with pytest.raises(FileNotFoundError, match='dataset_args.gsm8k.local_path'):
        adapter.load()
