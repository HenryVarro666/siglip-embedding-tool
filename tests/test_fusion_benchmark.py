import json
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from siglip_tools.benchmark import MODEL_REVISIONS
from siglip_tools.images import scan_images
from siglip_tools.index import ImageIndex


def fixture(tmp_path, alias="base"):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    root = tmp_path / "data"
    root.mkdir()
    rows = []
    for name, color in [("a", "red"), ("b", "blue"), ("c", "green")]:
        path = root / f"{name}.png"
        Image.new("RGB", (3, 2), color).save(path)
        rows.append(
            {
                "image_id": name,
                "path": path.name,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "source_page": f"https://commons.wikimedia.org/wiki/File:{name}.png",
                "download_url": f"https://upload.wikimedia.org/wikipedia/commons/{name}.png",
                "license_name": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "author": "Photographer",
            }
        )
    docs = {
        "manifest": {"dataset_id": "fixture", "images": rows},
        "queries": {
            "queries": [
                {"query_id": "i_zh", "intent_id": "i", "language": "zh", "text": "中文"},
                {"query_id": "i_en", "intent_id": "i", "language": "en", "text": "English"},
            ]
        },
        "qrels": {"qrels": {"i_zh": ["c"], "i_en": ["c"]}},
    }
    hashes = {}
    for name, document in docs.items():
        content = json.dumps(document).encode()
        (metadata / f"{name}.json").write_bytes(content)
        hashes[name] = sha256(content).hexdigest()
    protocol = {
        "experiment_id": "fixture",
        "dataset_id": "fixture",
        "dataset_hashes": hashes,
        "strategies": {
            "zh": {"languages": ["zh"], "weights": [1.0]},
            "en": {"languages": ["en"], "weights": [1.0]},
            "fusion": {"languages": ["zh", "en"], "weights": [0.5, 0.5]},
        },
        "device": "cpu",
        "dtype": "float32",
        "query_warmup": 1,
        "query_repeats": 2,
        "query_batch_sizes": {"zh": 1, "en": 1, "fusion": 2},
        "training": False,
        "weight_tuning": False,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    mid, rev = MODEL_REVISIONS[alias]
    identity = {"model_id": mid, "revision": rev, "runtime_versions": {"torch": torch.__version__}}
    index_path = tmp_path / "index.npz"
    ImageIndex([[1, 0], [0, 1], [0.6, 0.8]], scan_images(root), identity, root).save(index_path)
    return (
        SimpleNamespace(
            metadata=metadata,
            data_root=root,
            index=index_path,
            protocol=protocol_path,
            model=alias,
            output=tmp_path / "output",
            reference_result=None,
        ),
        protocol,
        identity,
    )


def test_complete_run_saves_three_strategies_once_per_intent(tmp_path, monkeypatch):
    from siglip_tools import fusion_benchmark as runner

    args, _, identity = fixture(tmp_path)

    class Encoder:
        device = torch.device("cpu")
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={"height": 224, "width": 224}),
            tokenizer=lambda texts, **kwargs: {"input_ids": [[1, 2] for text in texts]},
        )

        def __init__(self, *args, **kwargs):
            self.identity = identity
            assert kwargs["local_files_only"] is True

        def encode_texts(self, texts, batch_size=8):
            values = {"中文": [1, 0], "English": [0, 1]}
            return np.array([values[text] for text in texts], dtype=np.float32)

    monkeypatch.setattr(runner, "SigLIPEmbedder", Encoder)
    runner.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    assert result["independent_intents"] == 1
    assert set(result["strategies"]) == {"zh", "en", "fusion"}
    assert result["strategies"]["fusion"]["metrics"]["aggregate"]["average_precision"] == 1.0
    assert result["strategies"]["zh"]["metrics"]["aggregate"]["average_precision"] == 0.5
    assert result["strategies"]["en"]["metrics"]["aggregate"]["average_precision"] == 0.5
    assert result["strategies"]["fusion"]["rankings"]["i"][0]["score"] == pytest.approx(0.7)
    assert result["strategies"]["fusion"]["timing"]["count"] == 2
    assert result["diagnostics"]["i"]["text_query_cosine"] == 0
    assert result["training_executed"] is False
    for strategy in result["strategies"].values():
        assert strategy["rank_disagreements"] == 0
        assert list(strategy["metrics"]["per_query"]) == ["i"]


@pytest.mark.parametrize("change", ["gallery", "protocol_hash", "weights", "checkpoint"])
def test_bad_frozen_input_fails_before_model_allocation(tmp_path, monkeypatch, change):
    from siglip_tools import fusion_benchmark as runner

    args, protocol, _ = fixture(tmp_path)
    if change == "gallery":
        Image.new("RGB", (3, 2), "yellow").save(args.data_root / "a.png")
    elif change == "protocol_hash":
        protocol["dataset_hashes"]["queries"] = "changed"
    elif change == "weights":
        protocol["strategies"]["fusion"]["weights"] = [0.2, 0.8]
    else:
        args.model = "quality"
    args.protocol.write_text(json.dumps(protocol))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid fixed inputs must fail before allocating a model")

    monkeypatch.setattr(runner, "SigLIPEmbedder", forbidden)
    with pytest.raises(ValueError):
        runner.run_benchmark(args)


def test_index_records_must_match_frozen_gallery(tmp_path):
    from siglip_tools import fusion_benchmark as runner

    args, _, identity = fixture(tmp_path)
    index = ImageIndex.load(args.index)
    wrong_records = [type(r)(r.path, "f" * 64, r.pixel_sha256, r.aliases) for r in index.records]
    ImageIndex(index.vectors, wrong_records, identity, args.data_root).save(
        args.index, overwrite=True
    )
    with pytest.raises(ValueError, match="index|Index"):
        runner.load_inputs(args)


def test_compare_rejects_different_protocol_implementation_or_checkpoints(tmp_path):
    from siglip_tools.fusion_benchmark import compare_results

    def result(alias):
        mid, rev = MODEL_REVISIONS[alias]
        return {
            "model_alias": alias,
            "model_identity": {"model_id": mid, "revision": rev, "runtime_versions": {}},
            "dataset_hashes": {},
            "protocol_sha256": "p",
            "protocol": {},
            "implementation_sha256": "code",
            "environment": {},
            "strategies": {
                s: {"metrics": {"aggregate": {"average_precision": 0.5}}, "timing": {"p50": 10}}
                for s in ["zh", "en", "fusion"]
            },
        }

    base, quality = result("base"), result("quality")
    assert compare_results(base, quality)["quality_minus_base"]["fusion"]["average_precision"] == 0
    for key in ["protocol_sha256", "implementation_sha256", "environment"]:
        changed = {**quality, key: "changed"}
        with pytest.raises(ValueError):
            compare_results(base, changed)
    with pytest.raises(ValueError):
        compare_results(base, base)


def test_unchanged_ranks_do_not_hide_large_score_drift():
    from siglip_tools.fusion_benchmark import check_ranking_and_scores

    expected = [{"image_id": "a", "score": 0.1}, {"image_id": "b", "score": 0.0}]
    changed = [{"image_id": "a", "score": 0.9}, {"image_id": "b", "score": 0.0}]
    with pytest.raises(ValueError, match="score"):
        check_ranking_and_scores(changed, expected)
    with pytest.raises(ValueError, match="rank"):
        check_ranking_and_scores(list(reversed(expected)), expected)


def test_float32_scale_roundoff_is_recorded_within_tolerance():
    from siglip_tools.fusion_benchmark import check_ranking_and_scores

    expected = [{"image_id": "a", "score": 0.1}]
    rounded = [{"image_id": "a", "score": 0.10000001}]
    assert check_ranking_and_scores(rounded, expected) == pytest.approx(1e-8)


def test_historical_reference_rejects_same_rank_changed_scores(tmp_path):
    from siglip_tools import fusion_benchmark as runner

    reference = {
        "model_identity": {},
        "dataset_hashes": {},
        "implementation_sha256": runner.baseline_implementation_hash(),
        "rankings": {q: [{"image_id": "a", "score": 0.1}] for q in ["q_zh", "q_en"]},
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(reference))
    payload = {
        "model_identity": {},
        "dataset_hashes": {},
        "strategies": {
            s: {"rankings": {"i": [{"image_id": "a", "score": 0.9}]}} for s in ["zh", "en"]
        },
    }
    pairs = {"i": {lang: {"query_id": f"q_{lang}"} for lang in ["zh", "en"]}}
    with pytest.raises(ValueError, match="score"):
        runner._check_reference(path, payload, pairs)


def test_timed_rank_drift_does_not_publish_a_result(tmp_path, monkeypatch):
    from siglip_tools import fusion_benchmark as runner

    args, _, identity = fixture(tmp_path)

    class DriftingEncoder:
        device = torch.device("cpu")
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size={"height": 224, "width": 224}),
            tokenizer=lambda texts, **kwargs: {"input_ids": [[1] for text in texts]},
        )

        def __init__(self, *args, **kwargs):
            self.identity = identity
            self.chinese_calls = 0

        def encode_texts(self, texts, batch_size=8):
            values = {"中文": [1, 0], "English": [0, 1]}
            if texts == ["中文"]:
                self.chinese_calls += 1
                if self.chinese_calls >= 3:
                    values["中文"] = [0, 1]
            return np.array([values[text] for text in texts], dtype=np.float32)

    monkeypatch.setattr(runner, "SigLIPEmbedder", DriftingEncoder)
    with pytest.raises(ValueError, match="rank"):
        runner.run_benchmark(args)
    assert not (args.output / "result.json").exists()
