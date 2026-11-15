"""Static contracts for local benchmark dataset loading."""

import ast
import re
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parents[2] / 'evalscope' / 'benchmarks'
DIRECT_MODELSCOPE_APIS = {'MsDataset', 'dataset_file_download', 'dataset_snapshot_download'}


def _benchmark_sources():
    for path in BENCHMARK_ROOT.rglob('*.py'):
        yield path, ast.parse(path.read_text(encoding='utf-8'))


def test_benchmarks_do_not_download_modelscope_datasets() -> None:
    violations = []
    for path, tree in _benchmark_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'modelscope':
                imported = DIRECT_MODELSCOPE_APIS & {alias.name for alias in node.names}
                if imported:
                    violations.append(f'{path.relative_to(BENCHMARK_ROOT)}:{node.lineno}: {sorted(imported)}')
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.search(r'modelscope\.cn/datasets/.+/resolve/', node.value):
                    violations.append(f'{path.relative_to(BENCHMARK_ROOT)}:{node.lineno}: hard-coded resolve URL')

    assert not violations, 'Benchmark data must load through evalscope.api.dataset:\n' + '\n'.join(violations)
