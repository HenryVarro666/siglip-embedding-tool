import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from PIL import Image

from siglip_tools import benchmark
from siglip_tools.benchmark import compare_results, load_evaluation_data


def fixture_files(tmp_path):
    root = tmp_path / "data"
    (root / "images").mkdir(parents=True)
    images = []
    for name, color in (("a", "red"), ("b", "blue")):
        path = root / "images" / f"{name}.png"
        Image.new("RGB", (3, 2), color).save(path)
        images.append(
            {
                "image_id": name,
                "path": f"images/{name}.png",
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "source_page": f"https://commons.wikimedia.org/wiki/File:{name}.png",
                "download_url": f"https://upload.wikimedia.org/wikipedia/commons/{name}.png",
                "license_name": "CC BY-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "author": "Example photographer",
            }
        )
    documents = {
        "manifest": {"dataset_id": "fixture", "images": images},
        "queries": {
            "queries": [
                {"query_id": "q_en", "intent_id": "q", "language": "en", "text": "red device"},
                {"query_id": "q_zh", "intent_id": "q", "language": "zh", "text": "红色设备"},
            ]
        },
        "qrels": {"qrels": {"q_en": ["a"], "q_zh": ["a"]}},
    }
    paths = {}
    for name, content in documents.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(content), encoding="utf-8")
    return root, documents, paths


def load(paths, root):
    return load_evaluation_data(paths["manifest"], paths["queries"], paths["qrels"], root)


def test_load_freezes_hashes_and_preserves_declared_order(tmp_path):
    root, _, paths = fixture_files(tmp_path)
    data = load(paths, root)
    assert data is not None
    assert data["image_ids"] == ["a", "b"]
    assert [query["query_id"] for query in data["queries"]] == ["q_en", "q_zh"]
    assert data["hashes"]["qrels"] == sha256(paths["qrels"].read_bytes()).hexdigest()


def test_load_rejects_changed_image_content(tmp_path):
    root, _, paths = fixture_files(tmp_path)
    Image.new("RGB", (3, 2), "green").save(root / "images/a.png")
    with pytest.raises(ValueError, match="hash"):
        load(paths, root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("license_name", "CC BY-NC 4.0"),
        ("author", ""),
        ("download_url", "https://example.com/unknown.png"),
        ("path", "../outside.png"),
    ],
)
def test_load_requires_clear_license_attribution_and_safe_sources(tmp_path, field, value):
    root, documents, paths = fixture_files(tmp_path)
    documents["manifest"]["images"][0][field] = value
    paths["manifest"].write_text(json.dumps(documents["manifest"]))
    with pytest.raises(ValueError):
        load(paths, root)


def test_language_pairs_must_share_the_same_relevance_labels(tmp_path):
    root, documents, paths = fixture_files(tmp_path)
    documents["qrels"]["qrels"]["q_zh"] = ["b"]
    paths["qrels"].write_text(json.dumps(documents["qrels"]))
    with pytest.raises(ValueError, match="pair"):
        load(paths, root)


def test_qrels_cannot_name_an_unknown_image(tmp_path):
    root, documents, paths = fixture_files(tmp_path)
    documents["qrels"]["qrels"]["q_en"] = ["unknown"]
    paths["qrels"].write_text(json.dumps(documents["qrels"]))
    with pytest.raises(ValueError):
        load(paths, root)


def results():
    common = {
        "dataset_hashes": {"manifest": "a", "queries": "b", "qrels": "c"},
        "protocol": {
            "device": "mps",
            "dtype": "float32",
            "batch_size": 4,
            "query_warmup": 3,
            "query_repeats": 10,
        },
        "metrics": {"aggregate": {"hit_at_1": 0.5, "mrr": 0.75}},
        "timing": {"query_warm_ms": {"p50": 10.0}, "gallery_index_seconds": 2.0},
    }
    common["environment"] = {"machine": "arm64", "cpu_model": "M4 Pro"}
    common["implementation_sha256"] = "frozen-code"
    common["model_identity"] = {"runtime_versions": {"torch": "2.15.0"}}
    first = json.loads(json.dumps(common))
    second = json.loads(json.dumps(common))
    for alias, result in (("base", first), ("quality", second)):
        result["model_alias"] = alias
        model_id, revision = benchmark.MODEL_REVISIONS[alias]
        result["model_identity"].update(model_id=model_id, revision=revision)
    second["metrics"]["aggregate"]["hit_at_1"] = 0.75
    second["timing"]["query_warm_ms"]["p50"] = 20.0
    return first, second


def test_comparison_reports_delta_and_latency_ratio():
    base, quality = results()
    result = compare_results(base, quality)
    assert result is not None
    assert result["quality_minus_base"]["hit_at_1"] == 0.25
    assert result["quality_to_base_query_p50_ratio"] == 2.0


def test_comparison_rejects_different_dataset_or_device():
    for field in ("dataset_hashes", "protocol"):
        base, quality = results()
        if field == "dataset_hashes":
            quality[field]["qrels"] = "changed labels"
        else:
            quality[field]["device"] = "cpu"
        with pytest.raises(ValueError, match="same"):
            compare_results(base, quality)


@pytest.mark.parametrize("field", ["hardware", "runtime", "implementation"])
def test_comparison_rejects_changed_hardware_runtime_or_code(field):
    base, quality = results()
    if field == "hardware":
        quality["environment"]["cpu_model"] = "another machine"
    elif field == "runtime":
        quality["model_identity"]["runtime_versions"]["torch"] = "different"
    else:
        quality["implementation_sha256"] = "different"
    with pytest.raises(ValueError):
        compare_results(base, quality)


def test_comparison_requires_the_two_named_pinned_models():
    base, quality = results()
    with pytest.raises(ValueError):
        compare_results(base, base)
    quality["model_identity"]["revision"] = "another checkpoint"
    with pytest.raises(ValueError):
        compare_results(base, quality)


def test_parsed_labels_and_hash_come_from_the_same_bytes(tmp_path, monkeypatch):
    root, documents, paths = fixture_files(tmp_path)
    expected_hash = sha256(paths["qrels"].read_bytes()).hexdigest()
    original_text = type(paths["qrels"]).read_text
    original_bytes = type(paths["qrels"]).read_bytes
    changed = False

    def change_after_read(path, content):
        nonlocal changed
        if path == paths["qrels"] and not changed:
            changed = True
            documents["qrels"]["qrels"] = {"q_en": ["b"], "q_zh": ["b"]}
            path.write_text(json.dumps(documents["qrels"]))
        return content

    def read_text(path, *args, **kwargs):
        return change_after_read(path, original_text(path, *args, **kwargs))

    def read_bytes(path, *args, **kwargs):
        return change_after_read(path, original_bytes(path, *args, **kwargs))

    monkeypatch.setattr(type(paths["qrels"]), "read_text", read_text)
    monkeypatch.setattr(type(paths["qrels"]), "read_bytes", read_bytes)
    data = load(paths, root)
    assert changed
    assert data["qrels"]["q_en"] == ["a"]
    assert data["hashes"]["qrels"] == expected_hash


def test_changed_scan_baseline_is_rejected_before_model_load(tmp_path, monkeypatch):
    root, _, paths = fixture_files(tmp_path)
    original_scan = benchmark.scan_images

    def changed_scan(path):
        Image.new("RGB", (3, 2), "green").save(root / "images/a.png")
        return original_scan(path)

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Bad frozen inputs must fail before allocating a model")

    monkeypatch.setattr(benchmark, "scan_images", changed_scan)
    monkeypatch.setattr(benchmark, "SigLIPEmbedder", forbidden_load)
    args = SimpleNamespace(
        manifest=paths["manifest"],
        queries=paths["queries"],
        qrels=paths["qrels"],
        data_root=root,
        output=tmp_path / "output",
        batch_size=1,
        query_repeats=1,
        query_warmup=0,
        overwrite=False,
        model="base",
        device="cpu",
        local_files_only=True,
    )
    with pytest.raises(ValueError, match="hash"):
        benchmark.run_benchmark(args)


def test_complete_run_saves_transformers_size_object_as_json(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from transformers.image_utils import SizeDict

    root, _, paths = fixture_files(tmp_path)
    model_id, revision = benchmark.MODEL_REVISIONS["base"]

    class FakeEncoder:
        device = torch.device("cpu")
        identity = {"model_id": model_id, "revision": revision}
        processor = SimpleNamespace(
            image_processor=SimpleNamespace(size=SizeDict(height=224, width=224))
        )

        def __init__(self, *args, **kwargs):
            assert kwargs["local_files_only"] is True

        def encode_images(self, images, batch_size):
            return np.tile(np.array([[1, 0]], dtype=np.float32), (len(images), 1))

        def encode_texts(self, text):
            return np.array([[1, 0]], dtype=np.float32)

    monkeypatch.setattr(benchmark, "SigLIPEmbedder", FakeEncoder)
    args = SimpleNamespace(
        manifest=paths["manifest"],
        queries=paths["queries"],
        qrels=paths["qrels"],
        data_root=root,
        output=tmp_path / "output",
        batch_size=2,
        query_repeats=1,
        query_warmup=1,
        overwrite=False,
        model="base",
        device="cpu",
    )
    benchmark.run_benchmark(args)
    result = json.loads((args.output / "result.json").read_text())
    assert result["processor_image_size"] == {"height": 224, "width": 224}
    assert result["gallery_images"] == 2
    assert result["metrics"]["aggregate"]["hit_at_1"] == 1.0
    assert result["training_executed"] is False
    assert (args.output / "index.npz").is_file()
