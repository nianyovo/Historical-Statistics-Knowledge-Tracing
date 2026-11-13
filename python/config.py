from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


def positive(**values: float) -> None:
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "synthetic"
    path: str = "data/raw/skill_builder_data.csv"
    questions_path: str = "data/raw/EdNet/contents/questions.csv"
    csv_encoding: str = "utf-8-sig"
    split_seed: int = 42
    synthetic_seed: int = 1729
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    min_interactions: int = 5
    max_seq_len: int = 200
    chunk_protocol: str = "window_aligned"
    oof_folds: int = 5
    synthetic_students: int = 1200
    synthetic_min_length: int = 50
    synthetic_max_length: int = 350
    synthetic_items: int = 500
    synthetic_concepts: int = 40

    def __post_init__(self) -> None:
        if self.dataset not in {"synthetic", "assist2009", "assist2017", "ednet"}:
            raise ValueError("Unsupported dataset")
        if self.chunk_protocol not in {"window_aligned", "full_history_carry"}:
            raise ValueError("Unsupported chunk protocol")
        positive(train_ratio=self.train_ratio, val_ratio=self.val_ratio,
                 max_seq_len=self.max_seq_len, min_interactions=self.min_interactions,
                 synthetic_students=self.synthetic_students,
                 synthetic_items=self.synthetic_items, synthetic_concepts=self.synthetic_concepts)
        if self.train_ratio + self.val_ratio >= 1 or self.oof_folds < 2:
            raise ValueError("Invalid student split or OOF fold count")
        if not 5 <= self.synthetic_min_length <= self.synthetic_max_length:
            raise ValueError("Synthetic lengths must satisfy 5 <= min <= max")


@dataclass(frozen=True)
class StatConfig:
    window_size: int = 5
    prior_alpha: float = 10.0
    user_alpha: float = 2.0
    has_real_timestamp: bool = True
    has_real_resp_time: bool = True

    def __post_init__(self) -> None:
        positive(prior_alpha=self.prior_alpha, user_alpha=self.user_alpha)
        if not 1 <= self.window_size <= 1024:
            raise ValueError("window_size must be in [1, 1024]")


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.2

    def __post_init__(self) -> None:
        positive(hidden_dim=self.hidden_dim, num_heads=self.num_heads, num_layers=self.num_layers)
        if self.hidden_dim % self.num_heads or not 0 <= self.dropout < 1:
            raise ValueError("Invalid attention dimensions or dropout")


@dataclass(frozen=True)
class TrainConfig:
    seeds: tuple[int, ...] = (42, 2025, 2026)
    epochs: int = 30
    patience: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    device: str = "auto"
    cpu_threads: int = 4

    def __post_init__(self) -> None:
        positive(epochs=self.epochs, patience=self.patience, batch_size=self.batch_size,
                 learning_rate=self.learning_rate, grad_clip=self.grad_clip,
                 cpu_threads=self.cpu_threads)
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Provide distinct training seeds")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("Invalid weight decay")


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str = "outputs/formal"
    engine: str = "auto"
    cpp_library: str | None = None
    feature_threads: int = 4
    benchmark_sizes: tuple[int, ...] = (50000, 200000, 1000000)
    benchmark_threads: tuple[int, ...] = (1, 2, 4, 8)
    benchmark_repeats: int = 3

    def __post_init__(self) -> None:
        if self.engine not in {"auto", "cpp", "python"}:
            raise ValueError("engine must be auto, cpp or python")
        positive(feature_threads=self.feature_threads, benchmark_repeats=self.benchmark_repeats)
        if not self.benchmark_sizes or any(n < 10 for n in self.benchmark_sizes):
            raise ValueError("Benchmark sizes must be >= 10")
        if not self.benchmark_threads or any(n < 1 for n in self.benchmark_threads):
            raise ValueError("Benchmark thread counts must be positive")


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    stat: StatConfig = StatConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    experiment: ExperimentConfig = ExperimentConfig()


ConfigSection = TypeVar("ConfigSection")


def parse_section(cls: type[ConfigSection], values: dict[str, Any]) -> ConfigSection:
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} must be a mapping")
    hints = get_type_hints(cls)
    if set(values) - set(hints):
        raise ValueError(f"Unknown {cls.__name__} keys: {set(values) - set(hints)}")
    converted = dict(values)
    for name, value in values.items():
        expected = hints[name]
        if expected == tuple[int, ...]:
            if not isinstance(value, (list, tuple)) or any(type(x) is not int for x in value):
                raise ValueError(f"{name} must be an integer list")
            converted[name] = tuple(value)
        elif expected is float and type(value) in (int, float):
            converted[name] = float(value)
        elif expected == (str | None) and (value is None or isinstance(value, str)):
            continue
        elif type(value) is not expected:
            raise ValueError(f"Wrong type for {name}: expected {expected}")
    return cls(**converted)


def load_config(path: str | Path) -> Config:
    content = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Configuration must be a YAML mapping")
    section_types = get_type_hints(Config)
    if set(content) - set(section_types):
        raise ValueError("Unknown configuration section")
    sections = {f.name: parse_section(section_types[f.name], content.get(f.name, {}))
                for f in fields(Config)}
    return Config(**sections)
