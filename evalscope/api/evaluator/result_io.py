"""Parquet storage for native predictions and reviews."""

import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from filelock import FileLock

from evalscope.utils.logger import get_logger

logger = get_logger()

_JSON_COLUMNS = b'evalscope.json_columns'
_LAST_PART = b'evalscope.last_part'
_BATCH_SIZE = 512


def _result_schema(columns: list[tuple[str, pa.DataType]], *json_columns: str) -> pa.Schema:
    return pa.schema(
        [*columns, *((name, pa.large_string()) for name in json_columns)],
        metadata={_JSON_COLUMNS: json.dumps(json_columns)},
    )


PREDICTION_SCHEMA = _result_schema(
    [('index', pa.int64()), ('model', pa.string())], 'model_output', 'messages', 'agent_trace', 'metadata'
)
REVIEW_SCHEMA = _result_schema(
    [('index', pa.int64()), ('target', pa.list_(pa.string()))], 'messages', 'agent_trace', 'sample_score'
)


def _parts_dir(path: Path) -> Path:
    return path.with_name(f'.{path.name}.parts')


def _parts(path: Path) -> list[Path]:
    return sorted(_parts_dir(path).glob('*.parquet'))


def _part_number(path: Path) -> int:
    return int(path.stem)


def _last_part(path: Path) -> int:
    if path.is_file():
        return int((pq.read_metadata(path).metadata or {}).get(_LAST_PART, b'0'))
    return 0


@contextmanager
def _atomic_path(path: Path) -> Iterator[Path]:
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def result_rows_exist(path: str | Path) -> bool:
    """Check whether a subset has a final Parquet file or committed parts."""
    path = Path(path).expanduser()
    return path.suffix == '.parquet' and (path.is_file() or bool(_parts(path)))


def _decode_rows(parquet: pq.ParquetFile, path: Path, skip_invalid: bool) -> Iterator[dict[str, Any]]:
    columns = json.loads((parquet.schema_arrow.metadata or {}).get(_JSON_COLUMNS, b'[]'))
    for batch in parquet.iter_batches(batch_size=_BATCH_SIZE):
        for row in batch.to_pylist():
            try:
                for name in columns:
                    if row[name] is not None:
                        row[name] = json.loads(row[name])
            except (ValueError, TypeError, KeyError) as exc:
                if not skip_invalid:
                    raise
                logger.warning(f'Skipping invalid Parquet result row in {path}: {exc}')
                continue
            yield row


def read_result_rows(path: str | Path, *, skip_invalid: bool = False) -> list[dict[str, Any]]:
    """Read a native Parquet result file and any committed parts."""
    path = Path(path).expanduser()
    if path.suffix != '.parquet':
        raise ValueError(f'Evaluation result files must use the .parquet extension: {path}')

    # Snapshot the parts first. If compaction unlinks a part before it can be
    # opened, retry against the newly published file, whose footer covers it.
    for attempt in range(3):
        rows: list[dict[str, Any]] = []
        parts = _parts(path)
        try:
            last_part = 0
            if path.is_file():
                with pq.ParquetFile(path) as parquet:
                    last_part = int((parquet.metadata.metadata or {}).get(_LAST_PART, b'0'))
                    rows.extend(_decode_rows(parquet, path, skip_invalid))
            elif not parts:
                raise FileNotFoundError(f'No evaluation results at {path}')
            for part in parts:
                if _part_number(part) > last_part:
                    with pq.ParquetFile(part) as parquet:
                        rows.extend(_decode_rows(parquet, part, skip_invalid))
            return rows
        except FileNotFoundError:
            if attempt == 2:
                raise
    raise AssertionError('Unreachable result read state')


class ParquetResultWriter:
    """Write closed Parquet batches for resume and compact them into one result file."""

    def __init__(self, path: str | Path, schema: pa.Schema) -> None:
        self.path = Path(path).expanduser()
        self.schema = schema
        self._json_columns = json.loads(schema.metadata[_JSON_COLUMNS])
        self._rows: list[dict[str, Any]] = []

    def _encode(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            **row,
            **{
                name: json.dumps(row[name], ensure_ascii=False) if row.get(name) is not None else None
                for name in self._json_columns
            },
        }

    def write(self, row: dict[str, Any]) -> None:
        """Buffer a row and publish a readable checkpoint when the batch fills."""
        self._rows.append(self._encode(row))
        if len(self._rows) >= _BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        """Publish buffered rows as one complete, independently readable Parquet part."""
        if not self._rows:
            return
        with FileLock(self.path.parent / '.result-write.lock'):
            folder = _parts_dir(self.path)
            folder.mkdir(parents=True, exist_ok=True)
            previous = max([_last_part(self.path), *map(_part_number, _parts(self.path))])
            part = folder / f'{previous + 1:020d}.parquet'
            with _atomic_path(part) as temporary:
                pq.write_table(pa.Table.from_pylist(self._rows, schema=self.schema), temporary, compression='zstd')
            self._rows.clear()

    @staticmethod
    def _copy_parquet(path: Path, writer: pq.ParquetWriter) -> None:
        with pq.ParquetFile(path) as parquet:
            for batch in parquet.iter_batches(batch_size=_BATCH_SIZE):
                writer.write_batch(batch)

    def close(self, *, publish_to: str | Path | None = None) -> None:
        """Publish one atomic Parquet file, optionally replacing an earlier review."""
        self.flush()
        if publish_to is None and not _parts_dir(self.path).is_dir():
            return
        destination = Path(publish_to) if publish_to is not None else self.path
        with FileLock(destination.parent / '.result-write.lock'):
            self._compact(destination)

    def _compact(self, destination: Path) -> None:
        parts = _parts(self.path)
        last_published = _last_part(self.path)
        pending = [part for part in parts if _part_number(part) > last_published]
        if destination == self.path and not pending:
            self._remove_parts(self.path, parts)
            return

        replaced_parts = _parts(destination) if destination != self.path else []
        watermark = max([last_published, _last_part(destination), *map(_part_number, (*pending, *replaced_parts))])
        schema = self.schema.with_metadata({**self.schema.metadata, _LAST_PART: str(watermark)})
        with _atomic_path(destination) as temporary:
            with pq.ParquetWriter(temporary, schema, compression='zstd') as writer:
                if self.path.is_file():
                    self._copy_parquet(self.path, writer)
                for part in pending:
                    self._copy_parquet(part, writer)

        self._remove_parts(self.path, parts)
        if destination != self.path:
            self._remove_parts(destination, replaced_parts)

    def discard(self) -> None:
        """Discard unpublished transactional parts."""
        self._rows.clear()
        self._remove_parts(self.path, _parts(self.path))

    @staticmethod
    def _remove_parts(path: Path, parts: list[Path]) -> None:
        for part in parts:
            part.unlink(missing_ok=True)
        folder = _parts_dir(path)
        for temporary in folder.glob('.*.tmp'):
            temporary.unlink(missing_ok=True)
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
