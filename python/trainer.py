from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from python.config import Config, TrainConfig
from python.data_pipeline import KTDataset, PreparedData
from python.models import build_model


@dataclass(frozen=True)
class PredictionSet:
    labels: np.ndarray
    probabilities: np.ndarray
    original_step: np.ndarray
    local_step: np.ndarray
    frequency: np.ndarray
    user: np.ndarray
    student_length: np.ndarray
    stat_weight: np.ndarray


@dataclass(frozen=True)
class Evaluation:
    metrics: dict[str, float | None]
    counts: dict[str, int]
    predictions: PredictionSet


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    val_auc: float
    elapsed_seconds: float = 0.0


@dataclass
class EarlyStopping:
    patience: int
    best_auc: float = -math.inf
    best_epoch: int = 0
    stale_epochs: int = 0

    def update(self, auc: float, epoch: int, model: nn.Module, path: Path) -> bool:
        if not math.isfinite(auc):
            raise ValueError("Validation AUC is undefined; both classes are required")
        if auc > self.best_auc:
            self.best_auc, self.best_epoch, self.stale_epochs = auc, epoch, 0
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience


@dataclass(frozen=True)
class RunResult:
    model: str
    seed: int
    protocol: str
    ablation: str
    best_epoch: int
    best_val_auc: float
    test_evaluations: int
    metrics: dict[str, float | None]
    counts: dict[str, int]
    checkpoint: str
    predictions: str
    parameter_count: int


def seed_everything(seed: int, cpu_threads: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_device(cfg: TrainConfig) -> torch.device:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device if cfg.device == "auto" else cfg.device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; install a CUDA-enabled PyTorch build or set train.device=auto/cpu")
    return selected


def make_loader(dataset: KTDataset, cfg: TrainConfig, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle,
                      generator=generator, num_workers=0)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device,
               trim_padding: bool = False) -> dict[str, torch.Tensor]:
    length = batch["mask"].shape[1]
    if trim_padding:
        length = max(1, int(batch["mask"].sum(dim=1).max().item()))
    return {key: batch[key][:, :length].to(device) for key in ("items", "concepts", "labels", "stats", "mask")}


def forward_batch(model: nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    return model(batch["items"], batch["concepts"], batch["labels"], batch["stats"], batch["mask"])


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, grad_clip: float) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    for raw in loader:
        batch = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = forward_batch(model, batch)
        valid = batch["mask"]
        loss = nn.functional.binary_cross_entropy_with_logits(logits[valid], batch["labels"][valid].float())
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        count = int(valid.sum().item())
        total_loss += float(loss.item()) * count
        total_count += count
    if not total_count:
        raise ValueError("Empty training loader")
    return total_loss / total_count


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> PredictionSet:
    model.eval()
    names = list(PredictionSet.__dataclass_fields__)
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    with torch.no_grad():
        for raw in loader:
            # Evaluation has no Dropout. Keep training shapes and RNG consumption unchanged.
            batch = move_batch(raw, device, trim_padding=True)
            logits, gate = forward_batch(model, batch)
            valid = batch["mask"]
            probabilities = torch.sigmoid(logits[valid]).cpu().numpy()
            if not np.isfinite(probabilities).all():
                raise FloatingPointError("Non-finite predictions")
            collected["probabilities"].append(probabilities)
            for name in names:
                if name not in {"probabilities", "stat_weight"}:
                    collected[name].append(raw[name][raw["mask"]].numpy())
            weight = (1 - gate.mean(dim=-1))[valid].cpu().numpy() if gate is not None else np.full(len(probabilities), np.nan)
            collected["stat_weight"].append(weight)
    if not collected["labels"]:
        raise ValueError("Empty evaluation loader")
    return PredictionSet(**{name: np.concatenate(values) for name, values in collected.items()})


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probabilities))


def evaluate_predictions(pred: PredictionSet, max_length: int) -> Evaluation:
    y, p, step, freq = pred.labels, pred.probabilities, pred.original_step, pred.frequency
    metrics: dict[str, float | None] = {"overall_auc": safe_auc(y, p),
        "overall_acc": float(np.mean((p >= 0.5) == y)),
        "overall_rmse": float(np.sqrt(np.mean((p - y) ** 2)))}
    groups = {"step_1_5": step < 5, "step_6_10": (step >= 5) & (step < 10),
              "step_1_10": step < 10, "step_11_30": (step >= 10) & (step < 30),
              "step_gt_30": step >= 30, "freq_0_unseen": freq == 0,
              "freq_1_5_rare": (freq >= 1) & (freq <= 5),
              "freq_6_20_med": (freq >= 6) & (freq <= 20), "freq_gt_20_common": freq > 20,
              "long_student": pred.student_length > max_length,
              "long_student_later_blocks": (pred.student_length > max_length) & (step >= max_length)}
    counts = {"overall": len(y), "overall_positive": int(y.sum())}
    for name, selected in groups.items():
        metrics[f"{name}_auc"] = safe_auc(y[selected], p[selected])
        counts[name] = int(selected.sum())
        counts[f"{name}_positive"] = int(y[selected].sum())
    return Evaluation(metrics, counts, pred)


def save_training_state(directory: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                        loader: DataLoader, stopping: EarlyStopping, history: list[EpochRecord]) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "stopping": asdict(stopping), "history": [asdict(r) for r in history],
             "torch_rng": torch.get_rng_state(), "loader_rng": loader.generator.get_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = directory / "last.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(directory / "last.pt")
    (directory / "history.json").write_text(json.dumps(state["history"], indent=2), encoding="utf-8")


def restore_training_state(directory: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                           loader: DataLoader, patience: int) -> tuple[EarlyStopping, list[EpochRecord]]:
    if not (directory / "last.pt").exists():
        return EarlyStopping(patience), []
    state = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    loader.generator.set_state(state["loader_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return EarlyStopping(**state["stopping"]), [EpochRecord(**row) for row in state["history"]]


def fit_on_train_val(model: nn.Module, data: PreparedData, cfg: Config, seed: int,
                     directory: Path, device: torch.device) -> EarlyStopping:
    train_loader = make_loader(data.train, cfg.train, seed, True)
    val_loader = make_loader(data.val, cfg.train, seed, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                                 weight_decay=cfg.train.weight_decay)
    stopping, history = restore_training_state(directory, model, optimizer, train_loader, cfg.train.patience)
    for epoch in range(len(history) + 1, cfg.train.epochs + 1):
        if stopping.stale_epochs >= stopping.patience:
            break
        started = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, device, cfg.train.grad_clip)
        predictions = predict(model, val_loader, device)
        auc = safe_auc(predictions.labels, predictions.probabilities)
        if auc is None:
            raise ValueError("Validation split has only one class; no AUC-based selection possible")
        history.append(EpochRecord(epoch, loss, auc, time.perf_counter() - started))
        should_stop = stopping.update(auc, epoch, model, directory / "best.pt")
        save_training_state(directory, model, optimizer, train_loader, stopping, history)
        print(f"  epoch={epoch} train_loss={loss:.5f} val_auc={auc:.5f} seconds={history[-1].elapsed_seconds:.2f}", flush=True)
        if should_stop:
            break
    (directory / "history.json").write_text(json.dumps([asdict(r) for r in history], indent=2), encoding="utf-8")
    return stopping


def final_predictions(model: nn.Module, data: PreparedData, cfg: Config, seed: int,
                       directory: Path, device: torch.device) -> PredictionSet:
    path = directory / "test_predictions.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            return PredictionSet(**{key: archive[key] for key in PredictionSet.__dataclass_fields__})
    marker = directory / "test_started.json"
    if marker.exists():
        raise RuntimeError("Test was interrupted before caching; refusing an unrecorded repeat")
    marker.write_text('{"test_evaluations": 1}', encoding="utf-8")
    predictions = predict(model, make_loader(data.test, cfg.train, seed, False), device)
    temporary = directory / "test_predictions.tmp.npz"
    np.savez_compressed(temporary, **{key: getattr(predictions, key) for key in PredictionSet.__dataclass_fields__})
    temporary.replace(path)
    return predictions


def run_training(name: str, data: PreparedData, cfg: Config, seed: int,
                 directory: Path, ablation: str = "none", resume: bool = False) -> RunResult:
    directory.mkdir(parents=True, exist_ok=resume)
    if resume and (directory / "result.json").exists():
        result = RunResult(**json.loads((directory / "result.json").read_text(encoding="utf-8")))
        if not Path(result.predictions).is_file() or not Path(result.checkpoint).is_file():
            raise FileNotFoundError("Completed run is missing its evidence")
        return result
    seed_everything(seed, cfg.train.cpu_threads)
    device = get_device(cfg.train)
    model = build_model(name, len(data.vocabulary.items), len(data.vocabulary.concepts),
                        cfg.model, cfg.data.max_seq_len, ablation).to(device)
    stopping = fit_on_train_val(model, data, cfg, seed, directory, device)
    model.load_state_dict(torch.load(directory / "best.pt", map_location=device, weights_only=True))
    # The only Test inference call. All grouped metrics and gate plots reuse this cache.
    predictions = final_predictions(model, data, cfg, seed, directory, device)
    evaluation = evaluate_predictions(predictions, cfg.data.max_seq_len)
    prediction_path = directory / "test_predictions.npz"
    result = RunResult(name, seed, cfg.data.chunk_protocol, ablation, stopping.best_epoch,
                       stopping.best_auc, 1, evaluation.metrics, evaluation.counts,
                       str(directory / "best.pt"), str(prediction_path),
                       sum(parameter.numel() for parameter in model.parameters()))
    (directory / "result.json").write_text(json.dumps(asdict(result), indent=2, allow_nan=False), encoding="utf-8")
    return result
