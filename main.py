import os
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

_path_read_text = Path.read_text


def _read_text_utf8(self: Path, *args, **kwargs):
    if "encoding" not in kwargs or kwargs["encoding"] is None:
        kwargs["encoding"] = "utf-8"
    return _path_read_text(self, *args, **kwargs)


Path.read_text = _read_text_utf8 # Force UTF-8 partout, pour éviter les problèmes d'encodage sur Windows

import argparse

def main():
    parser = argparse.ArgumentParser(description="AI Safety GRPO pipeline")
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--n-prompts", type=int, default=100, help="Nombre de prompts pour eval")
    args = parser.parse_args()

    if args.mode == "train":
        from src.train import train
        train()
    elif args.mode == "eval":
        from src.eval import compute_asr
        compute_asr(n_prompts=args.n_prompts)


if __name__ == "__main__":
    main()
