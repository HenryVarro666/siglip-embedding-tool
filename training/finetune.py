"""Optional explicit fine-tuning entry point; never invoked by retrieval."""

import argparse
import json
from pathlib import Path

import torch

from siglip_tools.encoder import DEFAULT_MODEL, SigLIPEmbedder
from siglip_tools.images import file_sha256, load_rgb
from siglip_tools.losses import clip_loss, siglip_loss


def read_pairs(manifest: str | Path) -> list[tuple[Path, str]]:
    manifest = Path(manifest).resolve()
    pairs = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            image = Path(row["image"]).expanduser()
            if not image.is_absolute():
                image = manifest.parent / image
            text = row["text"]
            if not image.is_file() or not isinstance(text, str) or not text.strip():
                raise ValueError("Each pair requires an existing image and nonempty caption")
            pairs.append((image.resolve(), text))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid training manifest line {number}: {exc}") from exc
    if not pairs:
        raise ValueError("Training manifest has no image-caption pairs")
    return pairs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Optional pretrained SigLIP fine-tuning backup")
    parser.add_argument(
        "--pairs", required=True, type=Path, help="JSONL with image and text fields"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss", choices=["siglip", "clip"], default="siglip")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.batch_size < 2 or args.epochs < 1 or args.max_steps < 1 or args.learning_rate <= 0:
        parser.error("batch-size must be >=2; epochs, max-steps and learning-rate must be positive")
    if args.output.exists():
        parser.error("output already exists; use a new checkpoint directory")
    pairs = read_pairs(args.pairs)
    if len(pairs) < args.batch_size:
        parser.error("not enough pairs for one full batch; reduce batch-size (minimum 2)")
    torch.manual_seed(args.seed)
    encoder = SigLIPEmbedder(args.model, args.revision, args.device)
    tokenized = encoder.processor.tokenizer([text for _, text in pairs], truncation=False)
    if any(len(ids) > encoder.max_text_length for ids in tokenized["input_ids"]):
        parser.error(f"captions must fit the model's {encoder.max_text_length}-token limit")
    model = encoder.model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = siglip_loss if args.loss == "siglip" else clip_loss
    steps = 0
    last_loss = None
    for epoch in range(args.epochs):
        order = torch.randperm(len(pairs)).tolist()
        for start in range(0, len(order) - args.batch_size + 1, args.batch_size):
            batch = [pairs[i] for i in order[start : start + args.batch_size]]
            inputs = encoder.processor(
                images=[load_rgb(path) for path, _ in batch],
                text=[text for _, text in batch],
                padding="max_length",
                max_length=encoder.max_text_length,
                truncation=True,
                return_tensors="pt",
            ).to(encoder.device)
            optimizer.zero_grad(set_to_none=True)
            # Model logits already contain learned temperature and bias; do not add them twice.
            outputs = model(**inputs, return_loss=False)
            loss = loss_fn(outputs.logits_per_image)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss; checkpoint was not saved")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            last_loss = loss.detach().item()
            print(json.dumps({"epoch": epoch + 1, "step": steps, "loss": last_loss}))
            if steps >= args.max_steps:
                break
        if steps >= args.max_steps:
            break
    model.save_pretrained(args.output, safe_serialization=True)
    encoder.processor.save_pretrained(args.output)
    (args.output / "training_summary.json").write_text(
        json.dumps(
            {
                "steps": steps,
                "last_training_loss": last_loss,
                "loss": args.loss,
                "source_model_identity": encoder.identity,
                "manifest_sha256": file_sha256(args.pairs),
                "seed": args.seed,
                "evaluation": "not performed by this training entry point",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
