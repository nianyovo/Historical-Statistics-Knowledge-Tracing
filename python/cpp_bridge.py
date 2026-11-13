from __future__ import annotations

import ctypes as ct
import math
import weakref
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from python.config import StatConfig

NUM_STAT_FEATURES = 10
ROW_DTYPE = np.dtype([("user_id", "<i4"), ("item_id", "<i4"),
                      ("concept_id", "<i4"), ("correct", "<i4"),
                      ("timestamp_sec", "<f8"), ("resp_time_sec", "<f8")], align=True)


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
    concept_id: int
    correct: int
    timestamp_sec: float
    resp_time_sec: float


@dataclass(frozen=True)
class PackedBatch:
    rows: np.ndarray
    offsets: np.ndarray

    def __post_init__(self) -> None:
        if self.rows.dtype != ROW_DTYPE or self.rows.ndim != 1 or not self.rows.flags.c_contiguous:
            raise ValueError("rows must be a contiguous interaction array")
        if self.offsets.dtype != np.int64 or self.offsets.ndim != 1 or not self.offsets.flags.c_contiguous:
            raise ValueError("offsets must be contiguous int64")
        if not len(self.offsets) or self.offsets[0] != 0 or self.offsets[-1] != len(self.rows):
            raise ValueError("Invalid offsets")
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("Offsets must be monotonic")
        r = self.rows
        if np.any((r["correct"] != 0) & (r["correct"] != 1)):
            raise ValueError("Labels must be binary")
        if np.any(r["item_id"] == 0) or np.any(r["concept_id"] == 0):
            raise ValueError("ID zero is reserved for padding")
        if not np.isfinite(r["timestamp_sec"]).all() or not np.isfinite(r["resp_time_sec"]).all():
            raise ValueError("Times must be finite")
        if np.any(r["resp_time_sec"] < 0):
            raise ValueError("Response times must be nonnegative")
        for start, end in zip(self.offsets[:-1], self.offsets[1:]):
            if start < end and np.any(r["user_id"][start:end] != r["user_id"][start]):
                raise ValueError("A sequence must contain exactly one student")

    @property
    def num_seqs(self) -> int:
        return len(self.offsets) - 1


def pack_sequences(sequences: Sequence[Sequence[Interaction] | np.ndarray]) -> PackedBatch:
    offsets = np.zeros(len(sequences) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(s) for s in sequences])
    if sequences and all(isinstance(s, np.ndarray) for s in sequences):
        for sequence in sequences:
            if sequence.dtype != ROW_DTYPE or sequence.ndim != 1:
                raise ValueError("Array sequences must use ROW_DTYPE")
        root = sequences[0].base
        shared = isinstance(root, np.ndarray) and root.dtype == ROW_DTYPE and root.ndim == 1
        if shared and len(root) == offsets[-1] and all(
                s.base is root and s.ctypes.data == root.ctypes.data + int(offset) * ROW_DTYPE.itemsize
                for s, offset in zip(sequences, offsets[:-1])):
            return PackedBatch(root, offsets)
        return PackedBatch(np.concatenate(sequences), offsets)
    rows = np.empty(int(offsets[-1]), dtype=ROW_DTYPE)
    index = 0
    for sequence in sequences:
        for r in sequence:
            rows[index] = (r.user_id, r.item_id, r.concept_id, r.correct,
                           r.timestamp_sec, r.resp_time_sec)
            index += 1
    return PackedBatch(rows, offsets)


def validate_folds(batch: PackedBatch, folds: np.ndarray, num_folds: int) -> None:
    if folds.dtype != np.int32 or folds.shape != (batch.num_seqs,) or not folds.flags.c_contiguous:
        raise ValueError("fold_ids must be contiguous int32, one per sequence")
    if num_folds < 2 or set(folds.tolist()) != set(range(num_folds)):
        raise ValueError("Every OOF fold must be populated")
    user_folds: dict[int, int] = {}
    nonempty: set[int] = set()
    for start, end, fold in zip(batch.offsets[:-1], batch.offsets[1:], folds):
        if start == end:
            continue
        user = int(batch.rows[start]["user_id"])
        if user in user_folds and user_folds[user] != fold:
            raise ValueError("All chunks of one student must share an OOF fold")
        user_folds[user] = int(fold)
        nonempty.add(int(fold))
    if len(nonempty) != num_folds:
        raise ValueError("Empty OOF fold")


@dataclass
class Counts:
    count: int = 0
    correct: int = 0
    response_sum: float = 0.0

    def add(self, correct: int, response: float) -> None:
        self.count += 1
        self.correct += correct
        self.response_sum += response

    def minus(self, other: Counts) -> Counts:
        return Counts(self.count - other.count, self.correct - other.correct,
                      self.response_sum - other.response_sum)


@dataclass
class Aggregate:
    total: Counts = field(default_factory=Counts)
    items: dict[int, Counts] = field(default_factory=dict)
    concepts: dict[int, Counts] = field(default_factory=dict)

    def add(self, row: tuple[int, int, int, int, float, float]) -> None:
        _, item, concept, correct, _, response = row
        self.total.add(correct, response)
        self.items.setdefault(item, Counts()).add(correct, response)
        self.concepts.setdefault(concept, Counts()).add(correct, response)


def prior(total: Aggregate, excluded: Aggregate, item: int,
          concept: int, alpha: float) -> tuple[float, float, float, float]:
    global_counts = total.total.minus(excluded.total)
    mean = global_counts.correct / global_counts.count
    mean_rt = global_counts.response_sum / global_counts.count
    q = total.items.get(item, Counts()).minus(excluded.items.get(item, Counts()))
    c = total.concepts.get(concept, Counts()).minus(excluded.concepts.get(concept, Counts()))
    return (mean, (q.correct + alpha * mean) / (q.count + alpha),
            (c.correct + alpha * mean) / (c.count + alpha),
            (max(0.0, q.response_sum) + alpha * mean_rt) / (q.count + alpha))


def transform_sequence(rows: np.ndarray, total: Aggregate, excluded: Aggregate,
                       cfg: StatConfig, output: np.ndarray) -> None:
    history: dict[int, tuple[int, int, int, float]] = {}
    window: deque[int] = deque()
    correct_sum, streak, window_sum = 0, 0, 0
    for t, row in enumerate(rows.tolist()):
        _, item, concept, correct, timestamp, _ = row
        mean, qa, ca, rt = prior(total, excluded, item, concept, cfg.prior_alpha)
        n, c, last_step, last_ts = history.get(concept, (0, 0, 0, 0.0))
        gap = 0.0
        if n:
            gap = (math.log1p(min(max(timestamp - last_ts, 0), 2592000)) / 15
                   if cfg.has_real_timestamp else math.log1p(min(t - last_step, 500)) / 6.5)
        output[t] = [qa, ca, math.log1p(rt) / 6 if cfg.has_real_resp_time else 0,
                     (correct_sum + cfg.user_alpha * mean) / (t + cfg.user_alpha),
                     math.log1p(t) / 6, (c + cfg.user_alpha * ca) / (n + cfg.user_alpha),
                     math.log1p(n) / 5, window_sum / len(window) if window else mean,
                     math.tanh(streak / 5), gap]
        history[concept] = (n + 1, c + correct, t, timestamp)
        correct_sum += correct
        streak = max(streak, 0) + 1 if correct else min(streak, 0) - 1
        if len(window) == cfg.window_size:
            window_sum -= window.popleft()
        window.append(correct)
        window_sum += correct


class PurePythonStatEngine:
    def __init__(self, config: StatConfig = StatConfig()) -> None:
        self.config = config
        self.totals = Aggregate()

    def fit(self, batch: PackedBatch) -> None:
        if not len(batch.rows):
            raise ValueError("Empty training data")
        self.totals = Aggregate()
        for row in batch.rows.tolist():
            self.totals.add(row)

    def transform(self, batch: PackedBatch, num_threads: int = 1) -> np.ndarray:
        if not self.totals.total.count or num_threads < 1:
            raise ValueError("Fit engine first and use positive thread count")
        out = np.empty((len(batch.rows), NUM_STAT_FEATURES), dtype=np.float32)
        for start, end in zip(batch.offsets[:-1], batch.offsets[1:]):
            transform_sequence(batch.rows[start:end], self.totals, Aggregate(),
                               self.config, out[start:end])
        return out

    def fit_transform_oof(self, batch: PackedBatch, fold_ids: np.ndarray,
                          num_folds: int, num_threads: int = 1) -> np.ndarray:
        validate_folds(batch, fold_ids, num_folds)
        if num_threads < 1:
            raise ValueError("Thread count must be positive")
        self.totals = Aggregate()
        folds = [Aggregate() for _ in range(num_folds)]
        for start, end, fold in zip(batch.offsets[:-1], batch.offsets[1:], fold_ids):
            for row in batch.rows[start:end].tolist():
                self.totals.add(row)
                folds[fold].add(row)
        out = np.empty((len(batch.rows), NUM_STAT_FEATURES), dtype=np.float32)
        for start, end, fold in zip(batch.offsets[:-1], batch.offsets[1:], fold_ids):
            transform_sequence(batch.rows[start:end], self.totals, folds[fold],
                               self.config, out[start:end])
        return out

    def close(self) -> None:
        return None


class CConfig(ct.Structure):
    _fields_ = [("window_size", ct.c_size_t), ("prior_alpha", ct.c_double),
                ("user_alpha", ct.c_double), ("has_real_timestamp", ct.c_int32),
                ("has_real_resp_time", ct.c_int32)]


def check_status(status: int) -> None:
    if status:
        messages = {-1: "invalid input or unfitted engine", -2: "allocation failure", -3: "internal error"}
        raise RuntimeError(f"HSF C API: {messages.get(status, str(status))}")


def bind_library(path: Path) -> ct.CDLL:
    lib = ct.CDLL(str(path.resolve()))
    p, n = ct.c_void_p, ct.c_size_t
    signatures = {
        "hsf_create_engine": [ct.POINTER(CConfig), ct.POINTER(p)],
        "hsf_destroy_engine": [p], "hsf_fit_flat": [p, p, n, p, n],
        "hsf_transform_flat": [p, p, n, p, n, n, p, n],
        "hsf_fit_transform_oof_flat": [p, p, n, p, n, p, n, n, p, n],
    }
    for name, args in signatures.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = ct.c_int
    return lib


class CppStatEngine:
    def __init__(self, path: Path, config: StatConfig = StatConfig()) -> None:
        self.lib = bind_library(path)
        self.handle = ct.c_void_p()
        cfg = CConfig(config.window_size, config.prior_alpha, config.user_alpha,
                      int(config.has_real_timestamp), int(config.has_real_resp_time))
        check_status(self.lib.hsf_create_engine(ct.byref(cfg), ct.byref(self.handle)))
        self._finalizer = weakref.finalize(self, self.lib.hsf_destroy_engine, self.handle)

    def _args(self, batch: PackedBatch) -> tuple[object, ...]:
        if not self._finalizer.alive:
            raise RuntimeError("Engine is closed")
        return (self.handle, batch.rows.ctypes.data, len(batch.rows),
                batch.offsets.ctypes.data, batch.num_seqs)

    def fit(self, batch: PackedBatch) -> None:
        check_status(self.lib.hsf_fit_flat(*self._args(batch)))

    def transform(self, batch: PackedBatch, num_threads: int = 1) -> np.ndarray:
        if num_threads < 1:
            raise ValueError("Thread count must be positive")
        output = np.empty((len(batch.rows), NUM_STAT_FEATURES), dtype=np.float32)
        check_status(self.lib.hsf_transform_flat(*self._args(batch), num_threads,
                                                output.ctypes.data, output.size))
        return output

    def fit_transform_oof(self, batch: PackedBatch, fold_ids: np.ndarray,
                          num_folds: int, num_threads: int = 1) -> np.ndarray:
        validate_folds(batch, fold_ids, num_folds)
        if num_threads < 1:
            raise ValueError("Thread count must be positive")
        output = np.empty((len(batch.rows), NUM_STAT_FEATURES), dtype=np.float32)
        check_status(self.lib.hsf_fit_transform_oof_flat(
            *self._args(batch), fold_ids.ctypes.data, num_folds, num_threads,
            output.ctypes.data, output.size))
        return output

    def close(self) -> None:
        if self._finalizer.alive:
            check_status(self._finalizer())


StatEngine = PurePythonStatEngine | CppStatEngine


def find_library(explicit: str | None = None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "build", root / "build" / "Release"):
        for name in ("hsf_stat.dll", "libhsf_stat.dll", "libhsf_stat.so", "libhsf_stat.dylib"):
            if (directory / name).is_file():
                return directory / name
    return None


def create_engine(config: StatConfig, backend: str = "auto",
                  library: str | None = None) -> StatEngine:
    if backend not in {"auto", "cpp", "python"}:
        raise ValueError("Unknown engine backend")
    path = find_library(library) if backend != "python" else None
    if path:
        return CppStatEngine(path, config)
    if backend == "cpp":
        raise FileNotFoundError("Build hsf_stat first or specify cpp_library")
    return PurePythonStatEngine(config)
