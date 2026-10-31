"""Render local image/label and optional ranked-result review without running a model."""

import argparse
import json
from hashlib import sha256
from html import escape
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("benchmarks/equipment-v1"))
    parser.add_argument("--data-root", type=Path, default=Path("data/equipment-v1"))
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--fusion-result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("artifacts/equipment-review.html"))
    args = parser.parse_args()
    documents, hashes = {}, {}
    for name in ("manifest", "queries", "qrels"):
        content = (args.metadata / f"{name}.json").read_bytes()
        documents[name] = json.loads(content)
        hashes[name] = sha256(content).hexdigest()
    images = {r["image_id"]: r for r in documents["manifest"]["images"]}
    queries = documents["queries"]["queries"]
    qrels = documents["qrels"]["qrels"]
    results = [json.loads(path.read_text()) for path in args.result]
    fusion_results = [json.loads(path.read_text()) for path in args.fusion_result]
    for result in results + fusion_results:
        if result["dataset_hashes"] != hashes:
            raise ValueError("Result does not use these frozen labels/images")

    def card(image_id, relevant=None, rank=None, score=None):
        row = images[image_id]
        path = (args.data_root / row["path"]).resolve()
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed image: {image_id}")
        status = "" if relevant is None else (" · relevant" if relevant else " · negative")
        heading = image_id if rank is None else f"#{rank} {image_id}"
        number = "" if score is None else f" · cosine {score:.3f}"
        return (
            f'<figure><img src="{escape(path.as_uri())}" alt="{escape(row["title"])}">'
            f"<figcaption><b>{heading}{status}</b>{number}<br>"
            f"{escape(row['visual_description'])}<br>"
            f'<a href="{escape(row["source_page"])}">Source</a> · '
            f"{escape(row['author'])} · "
            f'<a href="{escape(row["license_url"])}">{escape(row["license_name"])}</a>'
            "</figcaption></figure>"
        )

    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Equipment retrieval pilot review</title><style>"
        "body{font:16px/1.5 system-ui;margin:24px;background:#f5f6f8;color:#182331}"
        "main{max-width:1500px;margin:auto}h1,h2{line-height:1.2}a{color:#125c98}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}"
        "figure{margin:0;padding:12px;background:white;border:1px solid #d9e0e6;border-radius:8px}"
        "figure img{width:100%;height:210px;object-fit:contain}figcaption{font-size:13px}"
        "section{margin:36px 0;border-top:2px solid #d9e0e6;padding-top:16px}"
        ".columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}"
        "code{word-break:break-all}</style><main><h1>40-image equipment retrieval pilot</h1>"
        "<p>Labels were frozen before scoring. AI-assisted visual/source annotation; "
        "not independent human ground truth. Relevant labels are binary source judgments, "
        "not model confidence. Images are local downloads, individually credited below.</p>"
    ]
    if fusion_results:
        automatic = "auto_en" in fusion_results[0]["strategies"]
        paired = {}
        for query in queries:
            paired.setdefault(query["intent_id"], {})[query["language"]] = query
        parts.append(
            "<p>Policies: direct Chinese / automatic English / Chinese+automatic English. "
            "Every automatic request retranslates; end-to-end timing includes translation.</p>"
            if automatic
            else "<p>Fixed policies: Chinese only / English only / equal mean cosine. "
            "English is supplied in advance; translation time is excluded.</p>"
        )
        for intent, pair in sorted(paired.items()):
            opened = " open" if intent in {"i04", "i05", "i07"} else ""
            parts.append(
                f"<section><details{opened}><summary><b>{intent}: "
                f"{escape(pair['zh']['text'])}</b></summary>"
                + ("" if automatic else f"<p>{escape(pair['en']['text'])}</p>")
                + '<div class="columns">'
            )
            for result in fusion_results:
                parts.append(f"<div><h2>{escape(result['model_alias'])}</h2>")
                if automatic:
                    parts.append(
                        f"<p>Automatic English: {escape(result['translations'][intent])}</p>"
                    )
                for strategy in (
                    ("zh", "auto_en", "auto_fusion") if automatic else ("zh", "en", "fusion")
                ):
                    policy = result["strategies"][strategy]
                    ap = policy["metrics"]["per_query"][intent]["average_precision"]
                    timing = policy["timing"].get("total_ms", policy["timing"])
                    parts.append(
                        f"<h3>{strategy} · AP {ap:.3f} · policy p50 {timing['p50']:.1f}ms</h3>"
                        '<div class="grid">'
                    )
                    for row in policy["rankings"][intent][:5]:
                        parts.append(
                            card(row["image_id"], row["relevant"], row["rank"], row["score"])
                        )
                    parts.append("</div>")
                parts.append("</div>")
            parts.append("</div></details></section>")
    elif results:
        for query in queries:
            qid = query["query_id"]
            parts.append(
                f"<section><h2>{qid}: {escape(query['text'])}</h2>"
                f'<p>Relevant gallery IDs: {", ".join(qrels[qid])}</p><div class="columns">'
            )
            for result in results:
                parts.append(f'<div><h3>{escape(result["model_alias"])}</h3><div class="grid">')
                for row in result["rankings"][qid][:5]:
                    parts.append(card(row["image_id"], row["relevant"], row["rank"], row["score"]))
                parts.append("</div></div>")
            parts.append("</div></section>")
    else:
        for query in queries:
            parts.append(
                f"<section><h2>{query['query_id']}: {escape(query['text'])}</h2>"
                '<p>Frozen positive images:</p><div class="grid">'
            )
            parts.extend(card(image_id, True) for image_id in qrels[query["query_id"]])
            parts.append("</div></section>")
    parts.append('<section><h2>Full gallery and attribution</h2><div class="grid">')
    parts.extend(card(image_id) for image_id in images)
    parts.append("</div></section></main></html>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
