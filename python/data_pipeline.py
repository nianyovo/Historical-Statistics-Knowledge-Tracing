from __future__ import annotations

import math
import re
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from python.config import Config, DataConfig, StatConfig
from python.cpp_bridge import ROW_DTYPE, Interaction, PackedBatch, create_engine, pack_sequences


@dataclass(frozen=True)
class RawInteraction:
    item: str
    concept: str
    correct: int
    timestamp: float
    response: float


@dataclass(frozen=True)
class RawColumns:
    item: np.ndarray
    concept: np.ndarray
    correct: np.ndarray
    timestamp: np.ndarray
    response: np.ndarray

    def __len__(self) -> int:
        return len(self.correct)

    def __getitem__(self, index: int) -> RawInteraction:
        return RawInteraction(str(self.item[index]), str(self.concept[index]), int(self.correct[index]),
                              float(self.timestamp[index]), float(self.response[index]))

    def __iter__(self) -> Iterator[RawInteraction]:
        for index in range(len(self)):
            yield self[index]


def raw_columns(rows: RawColumns | Sequence[RawInteraction]) -> RawColumns:
    if isinstance(rows, RawColumns):
        return rows
    return RawColumns(np.array([r.item for r in rows], dtype=object),
                      np.array([r.concept for r in rows], dtype=object),
                      np.fromiter((r.correct for r in rows), dtype=np.int32, count=len(rows)),
                      np.fromiter((r.timestamp for r in rows), dtype=np.float64, count=len(rows)),
                      np.fromiter((r.response for r in rows), dtype=np.float64, count=len(rows)))


@dataclass(frozen=True)
class StudentSequence:
    user: str
    rows: RawColumns | tuple[RawInteraction, ...]


@dataclass(frozen=True)
class DatasetBundle:
    sequences: tuple[StudentSequence, ...]
    has_real_timestamp: bool
    has_real_resp_time: bool
    source: str
    input_rows: int
    retained_rows: int
    cleaning_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class StudentSplit:
    train: tuple[StudentSequence, ...]
    val: tuple[StudentSequence, ...]
    test: tuple[StudentSequence, ...]


@dataclass(frozen=True)
class Vocabulary:
    items: dict[str, int]
    concepts: dict[str, int]


@dataclass(frozen=True)
class FeatureBlock:
    rows: np.ndarray
    stats: np.ndarray
    original_start: int
    student_length: int


@dataclass(frozen=True)
class PreparedData:
    train: KTDataset
    val: KTDataset
    test: KTDataset
    vocabulary: Vocabulary
    split: StudentSplit
    train_item_freq: dict[int, int]
    train_packed: PackedBatch
    train_folds: np.ndarray
    stat_config: StatConfig
    engine_name: str


def canonical_skill(value: str) -> str:
    if re.fullmatch(r"\d+(?:_\d+)+", str(value)):
        value = str(value).replace("_", ";")
    tokens = {s.strip() for s in re.split(r"[,;]", str(value)) if s.strip()}
    return ";".join(sorted(tokens))


def require_columns(frame: pd.DataFrame, names: Sequence[str]) -> None:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")


def parse_timestamps(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        if numeric.dropna().abs().median() > 100_000_000_000:
            numeric = numeric / 1000.0
        return numeric
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.map(lambda value: value.timestamp() if pd.notna(value) else np.nan)


def collapse_attempts(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if not frame.duplicated(keys).any():
        return frame
    grouped = frame.groupby(keys, sort=False, dropna=False)
    for column in ("item", "correct", "timestamp", "response"):
        if grouped[column].nunique().gt(1).any():
            raise ValueError(f"Conflicting duplicate attempt field: {column}")
    skills = grouped["concept"].agg(lambda values: canonical_skill(";".join(values)))
    unique = frame.drop_duplicates(keys, keep="first").set_index(keys)
    unique["concept"] = skills
    return unique.reset_index()


def clean_frame(frame: pd.DataFrame, real_timestamp: bool,
                real_response: bool, attempt_keys: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    audit = {"input_rows": len(frame)}
    for name in ("user", "item", "concept"):
        before = len(frame)
        frame[name] = frame[name].str.strip()
        frame = frame[frame[name].notna() & ~frame[name].str.lower().isin(["", "nan", "none", "-1", "noskill"])]
        audit[f"removed_missing_{name}"] = before - len(frame)
    before = len(frame)
    frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce")
    frame = frame[frame["correct"].isin([0, 1])]
    audit["removed_nonbinary_label"] = before - len(frame)
    frame["order"] = pd.to_numeric(frame["order"], errors="coerce")
    frame["timestamp"] = parse_timestamps(frame["timestamp"]) if real_timestamp else 0.0
    frame["response"] = pd.to_numeric(frame["response"], errors="coerce") if real_response else 0.0
    numeric = ["correct", "order", "timestamp", "response"]
    before = len(frame)
    frame = frame[np.isfinite(frame[numeric]).all(axis=1)].copy()
    audit["removed_invalid_numeric"] = before - len(frame)
    if real_response:
        audit["clipped_response_rows"] = int(((frame["response"] < 0.5) | (frame["response"] > 600)).sum())
        frame["response"] = frame["response"].clip(0.5, 600.0)
    frame["concept"] = frame["concept"].map(canonical_skill)
    before = len(frame)
    frame = collapse_attempts(frame, attempt_keys)
    audit["collapsed_duplicate_rows"] = before - len(frame)
    sort_keys = ["user", "timestamp", "order"] if real_timestamp else ["user", "order"]
    frame = frame.sort_values(sort_keys, kind="stable")
    if not real_timestamp:
        frame["timestamp"] = frame.groupby("user", sort=False).cumcount().astype(float)
    frame.attrs["cleaning_counts"] = audit
    return frame


def bundle_from_frame(frame: pd.DataFrame, cfg: DataConfig, real_ts: bool,
                      real_rt: bool, input_rows: int) -> DatasetBundle:
    sequences = []
    columns = ["item", "concept", "correct", "timestamp", "response"]
    for user, group in frame.groupby("user", sort=True):
        if len(group) < cfg.min_interactions:
            continue
        rows = RawColumns(*(group[column].to_numpy(dtype=dtype, copy=True)
                            for column, dtype in zip(columns, (object, object, np.int32, np.float64, np.float64))))
        sequences.append(StudentSequence(str(user), rows))
    if not sequences:
        raise ValueError("No students remain after cleaning")
    retained = sum(len(s.rows) for s in sequences)
    counts = dict(frame.attrs.get("cleaning_counts", {}))
    counts["removed_short_student_rows"] = len(frame) - retained
    counts["removed_short_students"] = int(frame["user"].nunique()) - len(sequences)
    return DatasetBundle(tuple(sequences), real_ts, real_rt, cfg.dataset, input_rows, retained, counts)


def load_assist(cfg: DataConfig) -> DatasetBundle:
    columns = {"user_id", "problem_id", "skill_id", "correct", "order_id", "ms_first_response"}
    if cfg.dataset != "assist2009":
        columns = {"studentId", "problemId", "skill", "skill_id", "correct", "startTime", "timeTaken"}
    raw = pd.read_csv(cfg.path, dtype=str, encoding=cfg.csv_encoding, low_memory=False,
                      usecols=lambda name: name in columns)
    if cfg.dataset == "assist2009":
        require_columns(raw, ["user_id", "problem_id", "skill_id", "correct", "order_id"])
        frame = raw.rename(columns={"user_id": "user", "problem_id": "item",
                                    "skill_id": "concept", "order_id": "order"}).copy()
        real_rt = "ms_first_response" in raw and pd.to_numeric(raw["ms_first_response"], errors="coerce").notna().any()
        frame["timestamp"] = "0"
        frame["response"] = pd.to_numeric(raw["ms_first_response"], errors="coerce") / 1000 if real_rt else 0.0
        keys, real_ts = ["user", "order"], False
    else:
        skill = "skill" if "skill" in raw else "skill_id"
        require_columns(raw, ["studentId", "problemId", skill, "correct", "startTime", "timeTaken"])
        frame = raw.rename(columns={"studentId": "user", "problemId": "item", skill: "concept",
                                    "startTime": "timestamp", "timeTaken": "response"}).copy()
        frame["order"] = np.arange(len(frame))
        keys, real_ts, real_rt = ["user", "item", "timestamp"], True, True
    frame = clean_frame(frame, real_ts, bool(real_rt), keys)
    return bundle_from_frame(frame, cfg, real_ts, bool(real_rt), len(raw))


def load_ednet(cfg: DataConfig) -> DatasetBundle:
    questions = pd.read_csv(cfg.questions_path, dtype=str,
                            usecols=lambda name: name in {"question_id", "correct_answer", "tags"})
    require_columns(questions, ["question_id", "correct_answer", "tags"])
    if questions["question_id"].duplicated().any():
        raise ValueError("Duplicate EdNet question metadata")
    files = sorted(Path(cfg.path).glob("u*.csv"))
    if not files:
        raise FileNotFoundError("No EdNet KT1 u*.csv files found")
    frames, input_rows = [], 0
    for path in files:
        raw = pd.read_csv(path, dtype=str, usecols=lambda name: name in
                          {"timestamp", "question_id", "user_answer", "elapsed_time"})
        require_columns(raw, ["timestamp", "question_id", "user_answer", "elapsed_time"])
        input_rows += len(raw)
        joined = raw.merge(questions[["question_id", "correct_answer", "tags"]],
                           on="question_id", how="inner", validate="many_to_one")
        joined = joined[joined["user_answer"].isin(list("abcd")) & joined["correct_answer"].isin(list("abcd"))]
        frame = pd.DataFrame({"user": path.stem, "item": joined["question_id"],
                              "concept": joined["tags"], "correct": (joined["user_answer"] == joined["correct_answer"]).astype(int),
                              "timestamp": pd.to_numeric(joined["timestamp"], errors="coerce") / 1000,
                              "response": pd.to_numeric(joined["elapsed_time"], errors="coerce") / 1000,
                              "order": np.arange(len(joined))})
        frames.append(clean_frame(frame, True, True, ["user", "item", "timestamp"]))
    return bundle_from_frame(pd.concat(frames, ignore_index=True), cfg, True, True, input_rows)


def synthetic_student(user: int, cfg: DataConfig, rng: np.random.Generator,
                      difficulties: np.ndarray) -> StudentSequence:
    length = int(rng.integers(cfg.synthetic_min_length, cfg.synthetic_max_length + 1))
    ability = float(rng.normal(0.2, 0.9))
    mastery: dict[int, float] = {}
    last_time: dict[int, float] = {}
    timestamp, rows = 1_700_000_000.0, []
    for _ in range(length):
        item = int(rng.integers(cfg.synthetic_items))
        concept = item % cfg.synthetic_concepts
        level = mastery.get(concept, ability)
        level -= 0.08 * math.log1p((timestamp - last_time.get(concept, timestamp)) / 3600)
        probability = 1 / (1 + math.exp(-max(-30, min(30, level - difficulties[item]))))
        correct = int(rng.random() < probability)
        response = float(np.clip(25 + 8 * difficulties[item] + rng.normal(0, 4), 0.5, 600))
        rows.append(RawInteraction(str(item), str(concept), correct, timestamp, response))
        mastery[concept] = level + (0.12 if correct else 0.04)
        last_time[concept] = timestamp
        timestamp += response + float(rng.uniform(20, 600))
    return StudentSequence(str(user), tuple(rows))


def load_dataset(cfg: DataConfig) -> DatasetBundle:
    if cfg.dataset == "ednet":
        return load_ednet(cfg)
    if cfg.dataset != "synthetic":
        return load_assist(cfg)
    rng = np.random.default_rng(cfg.synthetic_seed)
    difficulties = rng.normal(0, 1.1, cfg.synthetic_items)
    sequences = tuple(synthetic_student(u, cfg, rng, difficulties) for u in range(cfg.synthetic_students))
    n = sum(len(s.rows) for s in sequences)
    return DatasetBundle(sequences, True, True, "synthetic", n, n)


def split_students(bundle: DatasetBundle, cfg: DataConfig) -> StudentSplit:
    sequences = sorted((s for s in bundle.sequences if len(s.rows) >= cfg.min_interactions), key=lambda s: s.user)
    if len({s.user for s in sequences}) != len(sequences):
        raise ValueError("Dataset contains duplicate student sequences")
    order = np.random.default_rng(cfg.split_seed).permutation(len(sequences))
    shuffled = tuple(sequences[i] for i in order)
    n_train, n_val = int(len(shuffled) * cfg.train_ratio), int(len(shuffled) * cfg.val_ratio)
    if n_train < cfg.oof_folds or n_val < 1 or n_train + n_val >= len(shuffled):
        raise ValueError("Insufficient students for three-way split and OOF")
    return StudentSplit(shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:])


def build_vocabulary(train: Sequence[StudentSequence]) -> Vocabulary:
    columns = [raw_columns(s.rows) for s in train]
    items = np.unique(np.concatenate([c.item for c in columns]))
    concepts = np.unique(np.concatenate([c.concept for c in columns]))
    return Vocabulary({q: i + 1 for i, q in enumerate(items)},
                      {c: i + 1 for i, c in enumerate(concepts)})


def encode_ids(values: np.ndarray, vocabulary: dict[str, int]) -> np.ndarray:
    positions = pd.Index(list(vocabulary)).get_indexer(values)
    known = positions >= 0
    result = np.empty(len(values), dtype=np.int32)
    result[known] = np.array(list(vocabulary.values()), dtype=np.int32)[positions[known]]
    _, unknown = np.unique(values[~known], return_inverse=True)
    result[~known] = -unknown - 1
    return result


def encode_sequences(sequences: Sequence[StudentSequence], vocabulary: Vocabulary,
                     users: dict[str, int]) -> tuple[np.ndarray, ...]:
    columns = [raw_columns(s.rows) for s in sequences]
    lengths = np.array([len(s.rows) for s in sequences], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    rows = np.empty(int(offsets[-1]), dtype=ROW_DTYPE)
    if not columns:
        return ()
    rows["user_id"] = np.repeat([users[s.user] for s in sequences], lengths)
    rows["item_id"] = encode_ids(np.concatenate([c.item for c in columns]), vocabulary.items)
    rows["concept_id"] = encode_ids(np.concatenate([c.concept for c in columns]), vocabulary.concepts)
    rows["correct"] = np.concatenate([c.correct for c in columns])
    rows["timestamp_sec"] = np.concatenate([c.timestamp for c in columns])
    rows["resp_time_sec"] = np.concatenate([c.response for c in columns])
    return tuple(rows[start:end] for start, end in zip(offsets[:-1], offsets[1:]))


def engine_sequences(sequences: Sequence[np.ndarray], cfg: DataConfig) -> tuple[np.ndarray, ...]:
    if cfg.chunk_protocol == "full_history_carry":
        return tuple(sequences)
    return tuple(s[start:start + cfg.max_seq_len] for s in sequences
                 for start in range(0, len(s), cfg.max_seq_len))


def make_blocks(sequences: Sequence[np.ndarray], features: np.ndarray,
                max_length: int) -> tuple[FeatureBlock, ...]:
    blocks, cursor = [], 0
    for sequence in sequences:
        for start in range(0, len(sequence), max_length):
            rows = sequence[start:start + max_length]
            blocks.append(FeatureBlock(rows, features[cursor + start:cursor + start + len(rows)],
                                       start, len(sequence)))
        cursor += len(sequence)
    if cursor != len(features):
        raise ValueError("Feature row count mismatch")
    return tuple(blocks)


class KTDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, blocks: tuple[FeatureBlock, ...], vocabulary: Vocabulary,
                 item_frequency: dict[int, int], max_length: int) -> None:
        self.blocks = blocks
        self.vocabulary = vocabulary
        self.item_frequency = item_frequency
        self.max_length = max_length
        self._samples: dict[int, dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index not in self._samples:
            self._samples[index] = self._make_sample(index)
        return self._samples[index]

    def _make_sample(self, index: int) -> dict[str, torch.Tensor]:
        block = self.blocks[index]
        n, size = len(block.rows), self.max_length
        result = {key: torch.zeros(size, dtype=torch.long) for key in
                  ("items", "concepts", "labels", "original_step", "local_step", "frequency", "user", "student_length")}
        result["mask"] = torch.arange(size) < n
        result["stats"] = torch.zeros(size, 10, dtype=torch.float32)
        result["stats"][:n] = torch.from_numpy(block.stats)
        item_ids, concept_ids = block.rows["item_id"], block.rows["concept_id"]
        result["items"][:n] = torch.from_numpy(np.where(item_ids > 0, item_ids, len(self.vocabulary.items) + 1))
        result["concepts"][:n] = torch.from_numpy(np.where(concept_ids > 0, concept_ids, len(self.vocabulary.concepts) + 1))
        result["labels"][:n] = torch.from_numpy(block.rows["correct"].copy())
        result["original_step"][:n] = torch.arange(block.original_start, block.original_start + n)
        result["local_step"][:n] = torch.arange(n)
        result["frequency"][:n] = torch.tensor([self.item_frequency.get(int(q), 0) for q in item_ids])
        result["user"][:n] = int(block.rows[0]["user_id"])
        result["student_length"][:n] = block.student_length
        return result


def prepare_data(bundle: DatasetBundle, cfg: Config) -> PreparedData:
    split = split_students(bundle, cfg.data)
    vocabulary = build_vocabulary(split.train)
    users = {s.user: i for i, s in enumerate(sorted(bundle.sequences, key=lambda s: s.user))}
    encoded = [encode_sequences(part, vocabulary, users) for part in (split.train, split.val, split.test)]
    batches = [pack_sequences(engine_sequences(part, cfg.data)) for part in encoded]
    fold_order = np.random.default_rng(cfg.data.split_seed + 1).permutation(len(split.train))
    user_folds = {users[split.train[index].user]: i % cfg.data.oof_folds for i, index in enumerate(fold_order)}
    folds = np.array([user_folds[int(batches[0].rows[start]["user_id"])] for start in batches[0].offsets[:-1]], dtype=np.int32)
    stat_cfg = replace(cfg.stat, has_real_timestamp=bundle.has_real_timestamp,
                       has_real_resp_time=bundle.has_real_resp_time)
    with closing(create_engine(stat_cfg, cfg.experiment.engine, cfg.experiment.cpp_library)) as engine:
        train_stats = engine.fit_transform_oof(batches[0], folds, cfg.data.oof_folds, cfg.experiment.feature_threads)
        # OOF leaves the engine fitted to full Train; explicitly fit to make this protocol visible.
        engine.fit(batches[0])
        features = [train_stats] + [engine.transform(batch, cfg.experiment.feature_threads) for batch in batches[1:]]
        engine_name = type(engine).__name__
    ids, counts = np.unique(batches[0].rows["item_id"], return_counts=True)
    frequency = {int(q): int(n) for q, n in zip(ids, counts)}
    datasets = [KTDataset(make_blocks(part, stats, cfg.data.max_seq_len), vocabulary,
                          frequency, cfg.data.max_seq_len) for part, stats in zip(encoded, features)]
    return PreparedData(*datasets, vocabulary, split, frequency, batches[0], folds, stat_cfg, engine_name)
