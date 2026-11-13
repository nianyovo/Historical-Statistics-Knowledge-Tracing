from __future__ import annotations

import ctypes as ct
import math
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from python.config import Config, DataConfig, ModelConfig, StatConfig, load_config
from python.cpp_bridge import (CConfig, CppStatEngine, Interaction, PackedBatch,
                               PurePythonStatEngine, create_engine, find_library, pack_sequences)
from python.data_pipeline import (DatasetBundle, RawInteraction, StudentSequence,
                                  load_dataset, prepare_data)
from python.models import ABLATIONS, MODEL_NAMES, build_model
from python.trainer import PredictionSet, evaluate_predictions, run_training


@pytest.fixture(scope="session", autouse=True)
def limit_threads() -> None:
    torch.set_num_threads(2)


def example_sequences() -> list[list[Interaction]]:
    return [[Interaction(u, 1 + u * 12 + t, 1 + t % 3, (u + t) % 2,
                         100.0 * t, 10.0 + u + t / 3) for t in range(12)] for u in range(10)]


def engine_for(backend: str, cfg: StatConfig) -> PurePythonStatEngine | CppStatEngine:
    if backend == "cpp" and find_library() is None:
        pytest.skip("C++ library not built; run CMake to include native rigor checks")
    return create_engine(cfg, backend)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_oof_no_self_leakage(backend: str) -> None:
    sequences = example_sequences()
    folds = np.arange(10, dtype=np.int32) % 5
    changed = [[replace(r, correct=1 - r.correct, resp_time_sec=r.resp_time_sec + 100)
                for r in sequence] if i == 0 else sequence for i, sequence in enumerate(sequences)]
    with closing(engine_for(backend, StatConfig())) as engine:
        before = engine.fit_transform_oof(pack_sequences(sequences), folds, 5, 4)
        after = engine.fit_transform_oof(pack_sequences(changed), folds, 5, 4)
    np.testing.assert_array_equal(before[:12, :2], after[:12, :2])
    np.testing.assert_allclose(before[:12, 2], after[:12, 2], atol=1e-6, rtol=0)
    assert not np.array_equal(before[1:12, 3], after[1:12, 3])


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("real_ts,real_rt", [(True, True), (False, False), (False, True), (True, False)])
def test_oof_matches_independent_leave_fold_out(backend: str, real_ts: bool, real_rt: bool) -> None:
    sequences = example_sequences()
    cfg = StatConfig(has_real_timestamp=real_ts, has_real_resp_time=real_rt)
    folds = np.arange(10, dtype=np.int32) % 5
    with closing(engine_for(backend, cfg)) as engine:
        actual = engine.fit_transform_oof(pack_sequences(sequences), folds, 5, 4)
    for index, sequence in enumerate(sequences):
        reference = PurePythonStatEngine(cfg)
        reference.fit(pack_sequences([s for i, s in enumerate(sequences) if folds[i] != folds[index]]))
        expected = reference.transform(pack_sequences([sequence]))
        np.testing.assert_allclose(actual[index * 12:(index + 1) * 12], expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("real_ts,real_rt", [(True, True), (False, False), (False, True), (True, False)])
def test_cpp_python_oof_and_transform_parity(real_ts: bool, real_rt: bool) -> None:
    cfg = StatConfig(window_size=3, has_real_timestamp=real_ts, has_real_resp_time=real_rt)
    sequences = example_sequences()
    batch = pack_sequences(sequences)
    folds = np.arange(10, dtype=np.int32) % 5
    reference = PurePythonStatEngine(cfg)
    expected_oof = reference.fit_transform_oof(batch, folds, 5)
    held_out = pack_sequences([[replace(r, item_id=-1, concept_id=-7) for r in sequences[0]]])
    expected = reference.transform(held_out)
    with closing(engine_for("cpp", cfg)) as engine:
        for threads in (1, 4):
            actual = engine.fit_transform_oof(batch, folds, 5, threads)
            assert np.max(np.abs(actual - expected_oof)) < 1e-6
            engine.fit(batch)
            assert np.max(np.abs(engine.transform(held_out, threads) - expected)) < 1e-6


@pytest.mark.parametrize("backend", ["python", "cpp"])
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_causal_step_invariance(backend: str, model_name: str) -> None:
    sequences = example_sequences()
    batch = pack_sequences([sequences[0]])
    cfg = StatConfig()
    with closing(engine_for(backend, cfg)) as engine:
        engine.fit(pack_sequences(sequences[1:]))
        stats = engine.transform(batch)
        changed = batch.rows.copy()
        changed["correct"][5:] = 1 - changed["correct"][5:]
        changed["resp_time_sec"][5:] += 50
        changed_stats = engine.transform(PackedBatch(changed, batch.offsets))
    np.testing.assert_allclose(stats[:6], changed_stats[:6], atol=1e-6, rtol=0)
    torch.manual_seed(7)
    model = build_model(model_name, 120, 3, ModelConfig(hidden_dim=16, dropout=0), 12).eval()
    items = torch.tensor(batch.rows["item_id"].copy()).long().unsqueeze(0)
    concepts = torch.tensor(batch.rows["concept_id"].copy()).long().unsqueeze(0)
    labels = torch.tensor(batch.rows["correct"].copy()).long().unsqueeze(0)
    other_labels = torch.tensor(changed["correct"].copy()).long().unsqueeze(0)
    with torch.no_grad():
        before, _ = model(items, concepts, labels, torch.from_numpy(stats).unsqueeze(0))
        after, _ = model(items, concepts, other_labels, torch.from_numpy(changed_stats).unsqueeze(0))
    torch.testing.assert_close(torch.sigmoid(before[:, :6]), torch.sigmoid(after[:, :6]), atol=1e-6, rtol=0)


def tiny_config() -> Config:
    cfg = Config()
    return replace(cfg, data=replace(cfg.data, synthetic_students=30, synthetic_min_length=12,
                                     synthetic_max_length=16, max_seq_len=8, synthetic_items=25,
                                     synthetic_concepts=4),
                   model=ModelConfig(hidden_dim=16, num_heads=2, num_layers=2, dropout=0),
                   train=replace(cfg.train, epochs=2, patience=2, batch_size=8, seeds=(42,), cpu_threads=2))


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_window_aligned_fair_horizon(backend: str) -> None:
    if backend == "cpp" and find_library() is None:
        pytest.skip("C++ library not built")
    cfg = tiny_config()
    cfg = replace(cfg, experiment=replace(cfg.experiment, engine=backend))
    bundle = load_dataset(cfg.data)
    aligned = prepare_data(bundle, cfg)
    carried = prepare_data(bundle, replace(cfg, data=replace(cfg.data, chunk_protocol="full_history_carry")))
    for left, right in zip(aligned.test.blocks, carried.test.blocks):
        assert left.original_start == right.original_start
        np.testing.assert_array_equal(left.rows, right.rows)
        assert left.stats[0, 4] == 0
        if right.original_start:
            assert right.stats[0, 4] == pytest.approx(math.log1p(right.original_start) / 6)
            assert left.stats[0, 6] == 0
            assert left.stats[0, 8] == 0
            assert left.stats[0, 9] == 0
    sets = [set(s.user for s in part) for part in (aligned.split.train, aligned.split.val, aligned.split.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])


def test_train_only_vocabulary_and_distinct_unseen_concepts() -> None:
    sequences = tuple(StudentSequence(str(u), tuple(RawInteraction(f"q{u}", f"c{u}_{t % 2}", t % 2,
                                                                  float(t), 2.0) for t in range(8))) for u in range(20))
    data = prepare_data(DatasetBundle(sequences, False, True, "fixture", 160, 160), tiny_config())
    assert len(data.vocabulary.items) == len(data.split.train)
    sample = data.test[0]
    assert torch.all(sample["items"][sample["mask"]] == len(data.vocabulary.items) + 1)
    assert torch.all(sample["frequency"][sample["mask"]] == 0)
    stats = data.test.blocks[0].stats
    assert stats[1, 6] == 0
    assert stats[2, 6] > 0
    assert stats[0, 0] == stats[0, 3]


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_padding_and_backward(model_name: str) -> None:
    model = build_model(model_name, 5, 3, ModelConfig(hidden_dim=16, dropout=0), 8)
    items = torch.tensor([[1, 2, 6, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]])
    concepts = (items > 0).long()
    labels = items % 2
    stats = torch.rand(2, 8, 10)
    mask = items != 0
    logits, _ = model(items, concepts, labels, stats, mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], labels[mask].float())
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("group", list(ABLATIONS)[1:])
def test_ablation_removes_entire_group(group: str) -> None:
    model = build_model("HSF_Gate", 4, 2, ModelConfig(hidden_dim=16, dropout=0), 6, group).eval()
    ids, labels = torch.ones(2, 6, dtype=torch.long), torch.zeros(2, 6, dtype=torch.long)
    stats = torch.rand(2, 6, 10)
    changed = stats.clone()
    start, end = ABLATIONS[group]
    changed[:, :, start:end] += 50
    with torch.no_grad():
        before, _ = model(ids, ids, labels, stats)
        after, _ = model(ids, ids, labels, changed)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_controlled_models_share_initial_parameters() -> None:
    models = []
    for name in ("CausalSeqOnly", "HSF_Concat", "HSF_Gate"):
        torch.manual_seed(42)
        models.append(build_model(name, 10, 4, ModelConfig(hidden_dim=16), 12))
    states = [model.state_dict() for model in models]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
        assert torch.equal(states[0][key], states[2][key])
    for key in states[1]:
        assert torch.equal(states[1][key], states[2][key])


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_fold_integrity_and_empty_batch(backend: str) -> None:
    sequences = example_sequences()
    batch = pack_sequences(sequences)
    with closing(engine_for(backend, StatConfig())) as engine:
        engine.fit(batch)
        assert engine.transform(pack_sequences([])).shape == (0, 10)
        with pytest.raises((ValueError, RuntimeError)):
            engine.fit(pack_sequences([]))
        split = [sequences[0][:6], sequences[0][6:]] + sequences[1:]
        with pytest.raises(ValueError, match="student"):
            engine.fit_transform_oof(pack_sequences(split), np.arange(11, dtype=np.int32) % 5, 5)
    with pytest.raises((ValueError, RuntimeError)):
        engine.transform(batch, 0)


def test_c_abi_rejects_invalid_capacity_and_null_handles() -> None:
    with closing(engine_for("cpp", StatConfig())) as engine:
        batch = pack_sequences(example_sequences())
        engine.fit(batch)
        output = np.zeros(1, dtype=np.float32)
        assert engine.lib.hsf_transform_flat(*engine._args(batch), 1, output.ctypes.data, 1) == -1
        assert engine.lib.hsf_fit_flat(None, batch.rows.ctypes.data, len(batch.rows), batch.offsets.ctypes.data, batch.num_seqs) == -1
        handle = ct.c_void_p()
        cfg = CConfig(0, 10, 2, 1, 1)
        assert engine.lib.hsf_create_engine(ct.byref(cfg), ct.byref(handle)) == -1
        assert not handle.value


def test_assist2009_metadata_and_duplicate_attempts(tmp_path: Path) -> None:
    rows = [{"user_id": "u", "problem_id": str(i), "order_id": str(i), "skill_id": "1",
             "correct": i % 2, "ms_first_response": "2000"} for i in range(5)]
    rows.append(dict(rows[0], skill_id="2"))
    path = tmp_path / "assist.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    cfg = DataConfig(dataset="assist2009", path=str(path))
    bundle = load_dataset(cfg)
    assert not bundle.has_real_timestamp and bundle.has_real_resp_time
    assert len(bundle.sequences[0].rows) == 5
    assert bundle.sequences[0].rows[0].concept == "1;2"
    assert bundle.sequences[0].rows[1].timestamp == 1
    assert bundle.sequences[0].rows[0].response == 2
    pd.DataFrame(rows).drop(columns="ms_first_response").to_csv(path, index=False)
    assert not load_dataset(cfg).has_real_resp_time


def test_assist2017_real_time_sorting(tmp_path: Path) -> None:
    path = tmp_path / "assist2017.csv"
    rows = [{"studentId": "a", "problemId": str(i), "skill": "1", "correct": i % 2,
             "startTime": str(1_700_000_000 + i * 60), "timeTaken": "10"} for i in reversed(range(5))]
    pd.DataFrame(rows).to_csv(path, index=False)
    bundle = load_dataset(DataConfig(dataset="assist2017", path=str(path)))
    assert bundle.has_real_timestamp and bundle.has_real_resp_time
    assert bundle.sequences[0].rows[0].timestamp == 1_700_000_000
    assert bundle.sequences[0].rows[1].timestamp == 1_700_000_060


def test_ednet_units_and_labels(tmp_path: Path) -> None:
    questions = tmp_path / "questions.csv"
    pd.DataFrame([{"question_id": f"q{i}", "correct_answer": "a", "tags": "1;2"} for i in range(5)]).to_csv(questions, index=False)
    pd.DataFrame([{"timestamp": 1_700_000_000_000 + i * 60000, "question_id": f"q{i}",
                   "user_answer": "a" if i % 2 else "b", "elapsed_time": 1500} for i in range(5)]).to_csv(tmp_path / "u1.csv", index=False)
    bundle = load_dataset(DataConfig(dataset="ednet", path=str(tmp_path), questions_path=str(questions)))
    assert bundle.sequences[0].rows[0].timestamp == 1_700_000_000
    assert bundle.sequences[0].rows[0].response == 1.5
    assert [r.correct for r in bundle.sequences[0].rows] == [0, 1, 0, 1, 0]


def test_single_class_groups_are_missing_not_fabricated() -> None:
    pred = PredictionSet(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.7]), np.array([0, 5, 35]),
                         np.array([0, 5, 3]), np.array([0, 4, 50]), np.array([1, 1, 1]),
                         np.array([40, 40, 40]), np.full(3, np.nan))
    evaluation = evaluate_predictions(pred, 8)
    assert evaluation.metrics["overall_auc"] == 1
    assert evaluation.metrics["freq_0_unseen_auc"] is None
    assert evaluation.metrics["freq_6_20_med_auc"] is None
    assert evaluation.counts["step_1_5"] == 1
    assert evaluation.counts["long_student_later_blocks"] == 1


def test_training_uses_test_only_after_validation_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from python import trainer
    cfg = tiny_config()
    data = prepare_data(load_dataset(cfg.data), cfg)
    calls = []
    original = trainer.predict

    def spy(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
            device: torch.device) -> PredictionSet:
        calls.append(loader.dataset)
        if loader.dataset is data.test:
            assert (tmp_path / "run" / "best.pt").is_file()
        return original(model, loader, device)

    monkeypatch.setattr(trainer, "predict", spy)
    result = run_training("HSF_Gate", data, cfg, 42, tmp_path / "run")
    assert calls[-1] is data.test
    assert sum(dataset is data.test for dataset in calls) == 1
    assert all(dataset is data.val for dataset in calls[:-1])
    assert result.test_evaluations == 1 and result.best_epoch >= 1
    assert Path(result.predictions).is_file()


def test_config_rejects_unknown_keys_and_wrong_types(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    for text in ("data: {oof_folds: 1}", "model: {hidden_dim: 7, num_heads: 4}",
                 "train: {epochs: '4'}", "data: {typo: 1}", "stat: {prior_alpha: .nan}"):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


def test_resume_preserves_training_and_reuses_test_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from python import trainer
    cfg = tiny_config()
    cfg = replace(cfg, model=replace(cfg.model, dropout=0.2),
                   train=replace(cfg.train, device="cpu", epochs=3))
    data = prepare_data(load_dataset(cfg.data), cfg)
    uninterrupted = run_training("HSF_Gate", data, cfg, 42, tmp_path / "reference")
    original = trainer.train_epoch
    calls = 0

    def interrupt_epoch(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                        optimizer: torch.optim.Optimizer, device: torch.device, grad_clip: float) -> float:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("Simulated interruption after checkpoint")
        return original(model, loader, optimizer, device, grad_clip)

    monkeypatch.setattr(trainer, "train_epoch", interrupt_epoch)
    with pytest.raises(InterruptedError):
        run_training("HSF_Gate", data, cfg, 42, tmp_path / "resumed")
    monkeypatch.setattr(trainer, "train_epoch", original)
    resumed = run_training("HSF_Gate", data, cfg, 42, tmp_path / "resumed", resume=True)
    assert resumed.metrics == uninterrupted.metrics
    with np.load(resumed.predictions) as actual, np.load(uninterrupted.predictions) as expected:
        np.testing.assert_array_equal(actual["probabilities"], expected["probabilities"])

    def reject_inference(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                         device: torch.device) -> PredictionSet:
        raise AssertionError("Completed run must reuse cached result")

    monkeypatch.setattr(trainer, "predict", reject_inference)
    assert run_training("HSF_Gate", data, cfg, 42, tmp_path / "resumed", resume=True) == resumed


def test_dense_cpp_tables_preserve_sparse_and_negative_id_semantics() -> None:
    sequences = example_sequences()
    item_ids = [1, -3, 2147483647, 100000007]
    sequences = [[replace(r, item_id=item_ids[i % 4], concept_id=-17 if i % 2 else 2147483646)
                  for i, r in enumerate(s)] for s in sequences]
    batch, folds = pack_sequences(sequences), np.arange(10, dtype=np.int32) % 5
    reference = PurePythonStatEngine()
    expected = reference.fit_transform_oof(batch, folds, 5)
    with closing(engine_for("cpp", StatConfig())) as engine:
        np.testing.assert_array_equal(engine.fit_transform_oof(batch, folds, 5, 4), expected)
        np.testing.assert_array_equal(engine.transform(batch, 4), reference.transform(batch))


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_eval_trim_preserves_valid_outputs(model_name: str, device_name: str) -> None:
    from python.trainer import move_batch
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_name)
    torch.manual_seed(42)
    model = build_model(model_name, 10, 4, ModelConfig(hidden_dim=16, dropout=0.2), 40).to(device).eval()
    mask = torch.arange(40).unsqueeze(0) < torch.tensor([[7], [3], [1]])
    batch = {"items": torch.randint(1, 11, (3, 40)) * mask,
             "concepts": torch.randint(1, 5, (3, 40)) * mask,
             "labels": torch.randint(0, 2, (3, 40)), "stats": torch.rand(3, 40, 10), "mask": mask}
    full, trimmed = move_batch(batch, device), move_batch(batch, device, True)
    assert full["items"].shape == (3, 40)
    assert trimmed["items"].shape == (3, 7)
    with torch.no_grad():
        before, _ = model(**{k: full[k] for k in ("items", "concepts", "labels", "stats", "mask")})
        after, _ = model(**{k: trimmed[k] for k in ("items", "concepts", "labels", "stats", "mask")})
    torch.testing.assert_close(before[full["mask"]], after[trimmed["mask"]], atol=1e-6, rtol=0)


def test_columnar_packing_reuses_contiguous_storage() -> None:
    from python.data_pipeline import build_vocabulary, encode_sequences, engine_sequences, split_students
    cfg = tiny_config()
    bundle = load_dataset(cfg.data)
    split = split_students(bundle, cfg.data)
    vocabulary = build_vocabulary(split.train)
    encoded = encode_sequences(split.train, vocabulary, {s.user: i for i, s in enumerate(split.train)})
    packed = pack_sequences(engine_sequences(encoded, cfg.data))
    assert np.shares_memory(packed.rows, encoded[0])
    assert len(packed.rows) == sum(len(s.rows) for s in split.train)
