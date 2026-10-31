"""Compatibility wrapper for the packaged optional fine-tuning entry point."""

from siglip_tools.finetune import main

if __name__ == "__main__":
    raise SystemExit(main())
