from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import sys
import time
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from python.config import Config, load_config
from python.cpp_bridge import (CppStatEngine, PackedBatch, PurePythonStatEngine,
                               StatEngine, find_library)
from python.data_pipeline import DatasetBundle, PreparedData, load_dataset, prepare_data, raw_columns
from python.models import MODEL_NAMES
from python.trainer import RunResult, run_training


@dataclass(frozen=True)
class BenchmarkResult:
    interactions: int
    sequences: int
    backend: str
    threads: int
    median_ms: float
    min_ms: float
    max_ms: float
    speedup: float
    max_abs_error: float
    repeats: int


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def quick_config(cfg: Config) -> Config:
    return replace(cfg, data=replace(cfg.data, dataset="synthetic", synthetic_students=60,
                                     synthetic_min_length=12, synthetic_max_length=72,
                                     synthetic_items=80, synthetic_concepts=10, max_seq_len=32),
                   model=replace(cfg.model, hidden_dim=16, num_heads=2, num_layers=2, dropout=0.1),
                   train=replace(cfg.train, seeds=(42,), epochs=2, patience=2, batch_size=16),
                   experiment=replace(cfg.experiment, output_dir="outputs/quick", benchmark_sizes=(2000,),
                                      benchmark_repeats=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HSF-KT controlled experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="Synthetic smoke run, not thesis evidence")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--engine", choices=["auto", "cpp", "python"])
    parser.add_argument("--suite", choices=["all", "main", "benchmark"], default="all")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume identical config/code and reuse completed Test caches")
    return parser.parse_args()


def effective_config(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.quick:
        cfg = quick_config(cfg)
    cfg = replace(cfg, train=replace(cfg.train, seeds=tuple(args.seeds) if args.seeds else cfg.train.seeds,
                                     epochs=args.epochs if args.epochs is not None else cfg.train.epochs))
    return replace(cfg, experiment=replace(cfg.experiment, engine=args.engine or cfg.experiment.engine,
                                           output_dir=str(args.output) if args.output else cfg.experiment.output_dir))


def export_data_audit(data: PreparedData, bundle: DatasetBundle, output: Path) -> None:
    splits = {name: [s.user for s in getattr(data.split, name)] for name in ("train", "val", "test")}
    folds = {str(int(data.train_packed.rows[start]["user_id"])): int(fold)
             for start, fold in zip(data.train_packed.offsets[:-1], data.train_folds)}
    statistics = {name: {"students": len(getattr(data.split, name)),
                        "interactions": sum(len(s.rows) for s in getattr(data.split, name)),
                        "positive": sum(int(raw_columns(s.rows).correct.sum()) for s in getattr(data.split, name)),
                        "long_students": sum(len(s.rows) > data.train.max_length for s in getattr(data.split, name))}
                  for name in ("train", "val", "test")}
    write_json(output / "data_audit.json", {"source": bundle.source, "input_rows": bundle.input_rows,
        "retained_rows": bundle.retained_rows, "cleaning_counts": bundle.cleaning_counts,
        "split_statistics": statistics, "has_real_timestamp": bundle.has_real_timestamp,
        "has_real_resp_time": bundle.has_real_resp_time, "engine": data.engine_name,
        "split_users": splits, "train_fold_by_internal_user": folds,
        "item_vocabulary": data.vocabulary.items, "concept_vocabulary": data.vocabulary.concepts,
        "train_item_frequency": data.train_item_freq,
        "step_buckets": "original student step (zero-based internally); never local chunk step"})


def run_model_grid(data: PreparedData, cfg: Config, output: Path,
                   names: tuple[str, ...], ablations: tuple[str, ...] = ("none",), resume: bool = False) -> list[RunResult]:
    results = []
    for name in names:
        for ablation in ablations:
            for seed in cfg.train.seeds:
                directory = output / "runs" / f"{cfg.data.chunk_protocol}_{name}_{ablation}_s{seed}"
                print(f"[{name}] protocol={cfg.data.chunk_protocol} ablation={ablation} seed={seed}", flush=True)
                results.append(run_training(name, data, cfg, seed, directory, ablation, resume))
    return results


def run_model_experiments(bundle: DatasetBundle, aligned: PreparedData, cfg: Config,
                          output: Path, full_suite: bool, resume: bool = False) -> list[RunResult]:
    results = run_model_grid(aligned, cfg, output, MODEL_NAMES, resume=resume)
    write_json(output / "results.json", [asdict(result) for result in results])
    if full_suite:
        results.extend(run_model_grid(aligned, cfg, output, ("HSF_Gate",),
                                       ("item_prior", "user_cum", "recent_state"), resume))
        carry_cfg = replace(cfg, data=replace(cfg.data, chunk_protocol="full_history_carry"))
        carry = prepare_data(bundle, carry_cfg)
        results.extend(run_model_grid(carry, carry_cfg, output, ("HSF_Concat", "HSF_Gate"), resume=resume))
    write_json(output / "results.json", [asdict(result) for result in results])
    return results


def resize_benchmark(batch: PackedBatch, folds: np.ndarray, size: int) -> tuple[PackedBatch, np.ndarray]:
    by_fold = [np.flatnonzero(folds == fold).tolist() for fold in sorted(set(folds.tolist()))]
    order = [indices[i] for i in range(max(map(len, by_fold))) for indices in by_fold if i < len(indices)]
    rows = np.empty(size, dtype=batch.rows.dtype)
    offsets, out_folds, cursor, sequence = [0], [], 0, 0
    while cursor < size:
        index = order[sequence % len(order)]
        source = batch.rows[batch.offsets[index]:batch.offsets[index + 1]]
        take = min(len(source), size - cursor)
        rows[cursor:cursor + take] = source[:take]
        cursor += take
        offsets.append(cursor)
        out_folds.append(int(folds[index]))
        sequence += 1
    result_folds = np.array(out_folds, dtype=np.int32)
    if set(result_folds.tolist()) != set(folds.tolist()):
        raise ValueError("Benchmark size too small to represent every OOF fold")
    return PackedBatch(rows, np.array(offsets, dtype=np.int64)), result_folds


def measure_oof(engine: StatEngine, batch: PackedBatch, folds: np.ndarray,
                num_folds: int, threads: int, repeats: int) -> tuple[list[float], np.ndarray]:
    engine.fit_transform_oof(batch, folds, num_folds, threads)
    durations = []
    output = np.empty((0, 10), dtype=np.float32)
    for _ in range(repeats):
        start = time.perf_counter()
        output = engine.fit_transform_oof(batch, folds, num_folds, threads)
        durations.append((time.perf_counter() - start) * 1000)
    return durations, output


def benchmark_scale(data: PreparedData, cfg: Config, size: int, library: Path | None) -> list[BenchmarkResult]:
    batch, folds = resize_benchmark(data.train_packed, data.train_folds, size)
    repeats = cfg.experiment.benchmark_repeats
    reference = PurePythonStatEngine(data.stat_config)
    durations, expected = measure_oof(reference, batch, folds, cfg.data.oof_folds, 1, repeats)
    py_ms = statistics.median(durations)
    rows = [BenchmarkResult(size, batch.num_seqs, "python", 1, py_ms,
                            min(durations), max(durations), 1.0, 0.0, repeats)]
    if library:
        with closing(CppStatEngine(library, data.stat_config)) as engine:
            for threads in cfg.experiment.benchmark_threads:
                durations, actual = measure_oof(engine, batch, folds, cfg.data.oof_folds, threads, repeats)
                error = float(np.max(np.abs(actual - expected)))
                if error >= 1e-6:
                    raise AssertionError(f"C++/Python OOF parity failed: {error}")
                median = statistics.median(durations)
                rows.append(BenchmarkResult(size, batch.num_seqs, "cpp", threads, median,
                                             min(durations), max(durations), py_ms / median, error, repeats))
    return rows


def run_benchmarks(data: PreparedData, cfg: Config, output: Path) -> list[BenchmarkResult]:
    library = find_library(cfg.experiment.cpp_library)
    if not library:
        print("C++ library unavailable: benchmark contains Python only; speedups are not inferred.", flush=True)
    sizes = sorted(set((len(data.train_packed.rows), *cfg.experiment.benchmark_sizes)))
    results = []
    for size in sizes:
        print(f"[OOF benchmark] interactions={size}", flush=True)
        results.extend(benchmark_scale(data, cfg, size, library))
        write_json(output / "benchmarks.json", [asdict(result) for result in results])
    return results


def aggregate_results(results: list[RunResult]) -> list[dict[str, object]]:
    keys = sorted({(r.protocol, r.model, r.ablation) for r in results})
    aggregates = []
    for protocol, model, ablation in keys:
        group = [r for r in results if (r.protocol, r.model, r.ablation) == (protocol, model, ablation)]
        metrics = {}
        for name in group[0].metrics:
            values = [r.metrics[name] for r in group if r.metrics[name] is not None]
            metrics[name] = {"mean": float(np.mean(values)) if values else None,
                             "std": float(np.std(values, ddof=1)) if len(values) > 1 else (0.0 if values else None),
                             "n_seeds": len(values)}
        aggregates.append({"protocol": protocol, "model": model, "ablation": ablation, "metrics": metrics})
    return aggregates


def format_metric(metric: dict[str, float | int | None]) -> str:
    if metric["mean"] is None:
        return "N/A"
    return f"{metric['mean']:.4f} ± {metric['std']:.4f}"


def export_tables(results: list[RunResult], benchmarks: list[BenchmarkResult], output: Path) -> None:
    aggregated = aggregate_results(results)
    write_json(output / "summary.json", aggregated)
    columns = ["overall_auc", "overall_acc", "overall_rmse", "step_1_5_auc", "step_1_10_auc"]
    lines = ["# HSF-KT experiment results", "", "Mean ± sample SD over training seeds. N/A = empty or single-class group.",
             "A one-seed SD of 0 is not evidence of stability. Synthetic/quick runs are pipeline checks.", "",
             "| Protocol | Model | Ablation | " + " | ".join(columns) + " |",
             "| --- | --- | --- | " + " | ".join(["---"] * len(columns)) + " |"]
    latex = [r"\begin{tabular}{lllrrrrr}", r"\toprule", r"Protocol & Model & Ablation & AUC & ACC & RMSE & AUC$_{1:5}$ & AUC$_{1:10}$ \\", r"\midrule"]
    for row in aggregated:
        values = [str(row[key]) for key in ("protocol", "model", "ablation")]
        values += [format_metric(row["metrics"][column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
        latex.append(" & ".join(v.replace("_", r"\_").replace("±", r"$\pm$") for v in values) + r" \\")
    latex += [r"\bottomrule", r"\end{tabular}"]
    append_group_tables(lines, aggregated)
    lines += ["", "## OOF engine benchmark", "", "| Rows | Backend | Threads | Median ms | Speedup | Max error |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {b.interactions} | {b.backend} | {b.threads} | {b.median_ms:.3f} | {b.speedup:.2f} | {b.max_abs_error:.3g} |" for b in benchmarks]
    (output / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "tables.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def append_group_tables(lines: list[str], rows: list[dict[str, object]]) -> None:
    groups = {"History and item rarity": ["step_1_5_auc", "step_6_10_auc", "step_11_30_auc", "step_gt_30_auc",
                                         "freq_0_unseen_auc", "freq_1_5_rare_auc", "freq_6_20_med_auc", "freq_gt_20_common_auc"],
              "Long-student context attribution": ["long_student_auc", "long_student_later_blocks_auc"]}
    for title, columns in groups.items():
        lines.extend(["", f"## {title}", "", "| Protocol / model / ablation | " + " | ".join(columns) + " |",
                      "| --- | " + " | ".join(["---"] * len(columns)) + " |"])
        for row in rows:
            identity = " / ".join(str(row[k]) for k in ("protocol", "model", "ablation"))
            lines.append("| " + identity + " | " + " | ".join(format_metric(row["metrics"][c]) for c in columns) + " |")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(cfg: Config, args: argparse.Namespace, output: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sources = list((root / "python").glob("*.py")) + list((root / "cpp").rglob("*.cpp"))
    sources += list((root / "cpp").rglob("*.hpp")) + list((root / "cpp").rglob("*.h"))
    sources += [root / "CMakeLists.txt", root / "requirements.txt", root / "requirements-lock.txt"]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    data_hash = file_sha256(Path(cfg.data.path)) if cfg.data.dataset != "synthetic" and Path(cfg.data.path).is_file() else None
    manifest = {"python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": np.__version__, "cuda_available": torch.cuda.is_available(),
        "processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "data_sha256": data_hash,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "quick_smoke_only": args.quick, "suite": args.suite, "source_sha256": hashes,
        "benchmark_scope": "Warm-up then repeated OOF fit + transform; prepacked input; allocation and thread startup included; no CSV/model training.",
        "benchmark_expansion": "Input sequences repeated/truncated with original student fold retained; performance measurements only.",
        "uncertainty": "Sample SD across training seeds; fixed data/split/OOF seeds; no significance claim."}
    if args.resume:
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if previous["source_sha256"] != hashes or load_config(output / "config.yaml") != cfg:
            raise ValueError("Resume requires identical source files and configuration")
        if previous["torch"] != torch.__version__ or previous["suite"] != args.suite:
            raise ValueError("Resume requires the same Torch version and experiment suite")
        if previous.get("data_sha256") != data_hash:
            raise ValueError("Dataset file changed since this experiment was started")
        return
    write_json(output / "manifest.json", manifest)
    (output / "config.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = effective_config(args)
    if cfg.data.chunk_protocol != "window_aligned":
        raise ValueError("The controlled suite must start with window_aligned; all includes carry experiments automatically")
    output = Path(cfg.experiment.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite experiment evidence: choose a new --output ({output})")
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(cfg, args, output)
    bundle = load_dataset(cfg.data)
    data = prepare_data(bundle, cfg)
    export_data_audit(data, bundle, output)
    results = [] if args.suite == "benchmark" else run_model_experiments(bundle, data, cfg, output, args.suite == "all", args.resume)
    benchmarks = run_benchmarks(data, cfg, output) if args.suite != "main" else []
    export_tables(results, benchmarks, output)
    if not args.no_plots:
        from python.plot_figures import plot_all
        plot_all(output)
    print(f"Completed: {output}", flush=True)


if __name__ == "__main__":
    main()
