# Publication copy: comments/line endings only; algorithm AST unchanged.
# Project migration copy; retain upstream method and host notices. No experiment executed.
# Component licensing and exact source hashes: sources/FILE_DISPOSITIONS.json.
"""Durable, single-writer result journal, independent of any GPU evaluator.

One record must contain one COMPLETE request (sample ID x mode), including its
answer, timings and execution-epoch identity. Paired analysis joins committed
records later; do not delay saving until every mode finishes. Call commit OUTSIDE
measured boundaries.
Reopen the same directory and semantic contract to resume. No records are
overwritten. An unterminated tail is retained and reported, never truncated.

This is NOT a SmolVLM runner. The final runner still needs real interruption
testing on its target persistent filesystem. Durability requires that filesystem
to honor fsync and process locks; this module cannot protect against disk loss.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid
import zlib


class JournalError(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sync_directory(path):
    # Windows does not expose a portable directory-fsync through os.open.
    # Linux deployment must execute this branch successfully (no silent ignore).
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Journal:
    def __init__(self, directory, contract, *, segment_records=1024):
        if segment_records < 1:
            raise ValueError("segment_records must be positive")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        sync_directory(self.directory.parent)
        self._lock = None
        self._stream = None
        self._failed = False
        self._segment_limit = segment_records
        self._segment_count = 0
        self._records = {}
        self.incomplete_tails = []
        # Take a JSON snapshot so caller mutations cannot alter the contract.
        self.contract = json.loads(canonical(contract))
        try:
            self._take_lock()
            self._check_contract()
            self._load()
        except BaseException:
            self.close()
            raise

    def _take_lock(self):
        self._lock = (self.directory / "writer.lock").open("a+b")
        # Lock file persists; ownership is an OS lock, not a stale PID marker.
        if os.fstat(self._lock.fileno()).st_size == 0:
            self._lock.write(b"0")
            self._lock.flush()
        self._lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self._lock.close()
            self._lock = None
            raise JournalError("result directory already has a writer") from error

    def _check_contract(self):
        path = self.directory / "contract.json"
        if path.exists():
            if json.loads(path.read_bytes()) != self.contract:
                raise JournalError("contract changed: use a new result directory")
            return
        if any(self.directory.glob("segment_*.jsonl")):
            raise JournalError("segments exist but contract is missing")
        temporary = self.directory / ("contract." + uuid.uuid4().hex + ".pending")
        with temporary.open("xb") as stream:
            stream.write(canonical(self.contract) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(self.directory)

    def _load(self):
        segments = sorted(self.directory.glob("segment_*.jsonl"))
        ids = [int(path.stem.split("_")[1]) for path in segments]
        if ids != list(range(len(ids))):
            raise JournalError("missing or nonconsecutive journal segment")
        for path in segments:
            with path.open("rb") as stream:
                offset = 0
                for line_number, line in enumerate(stream, 1):
                    if not line.endswith(b"\n"):
                        self.incomplete_tails.append({
                            "file": path.name, "offset": offset, "bytes": len(line)})
                        break
                    try:
                        envelope = json.loads(line)
                        record = envelope["record"]
                        expected = f"{zlib.crc32(canonical(record)):08x}"
                        if envelope["crc32"] != expected:
                            raise ValueError("CRC mismatch")
                        key = record["key"]
                        if not isinstance(key, str) or not key:
                            raise ValueError("invalid unit key")
                        if not isinstance(record["result"], dict):
                            raise ValueError("invalid result")
                        if key in self._records:
                            raise ValueError("duplicate committed unit")
                        self._records[key] = record["result"]
                    except (ValueError, KeyError, TypeError, UnicodeError) as error:
                        raise JournalError(
                            f"corrupt committed record: {path.name}:{line_number}"
                        ) from error
                    offset += len(line)
        # Always start a NEW segment, including after a preserved torn tail.
        self._next_segment = len(segments)

    def _start_segment(self):
        if self._stream is not None:
            self._stream.close()
        path = self.directory / f"segment_{self._next_segment:06d}.jsonl"
        self._stream = path.open("xb")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        sync_directory(self.directory)
        self._next_segment += 1
        self._segment_count = 0

    def contains(self, key):
        return key in self._records

    def result(self, key):
        return json.loads(canonical(self._records[key]))

    def committed_keys(self):
        return frozenset(self._records)

    def missing(self, expected_keys):
        expected_keys = list(expected_keys)
        if len(expected_keys) != len(set(expected_keys)):
            raise JournalError("expected manifest contains duplicate unit keys")
        unexpected = self.committed_keys() - set(expected_keys)
        if unexpected:
            raise JournalError("journal contains units absent from expected manifest")
        return [key for key in expected_keys if key not in self._records]

    def commit(self, key, result):
        if self._lock is None or self._failed:
            raise JournalError("journal closed or failed; reopen before continuing")
        if not isinstance(key, str) or not key or not isinstance(result, dict):
            raise ValueError("a nonempty unit key and result dictionary are required")
        result = json.loads(canonical(result))
        if key in self._records:
            if self._records[key] == result:
                return False
            raise JournalError("refusing to overwrite a committed result")
        record = {"key": key, "result": result}
        envelope = {"record": record,
                    "crc32": f"{zlib.crc32(canonical(record)):08x}"}
        try:
            if self._stream is None or self._segment_count >= self._segment_limit:
                self._start_segment()
            self._stream.write(canonical(envelope) + b"\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            # An uncertain write must be re-read, not blindly retried in place.
            self._failed = True
            raise
        self._records[key] = result
        self._segment_count += 1
        return True

    def close(self):
        try:
            if self._stream is not None:
                self._stream.close()
                self._stream = None
        finally:
            if self._lock is not None:
                # Closing the handle releases the process lock even on failure.
                self._lock.close()
                self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
