import unsloth  # must be first before peft/transformers

import msvcrt
import sys
import threading
import time
from pathlib import Path

from safetensors.torch import load_file
from transformers import StoppingCriteria, StoppingCriteriaList, TextStreamer
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).parent))
from model import GemmaModel, MAX_SEQ_LENGTH

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "outputs"


class _StopFlag:
    def __init__(self):
        self._v = False

    def set(self):
        self._v = True

    def clear(self):
        self._v = False

    def is_set(self):
        return self._v


class _KeyStopCriteria(StoppingCriteria):
    def __init__(self, flag: _StopFlag):
        self._flag = flag

    def __call__(self, input_ids, scores, **kwargs):
        return self._flag.is_set()


def _last_checkpoint() -> Path:
    checkpoints = sorted(
        OUTPUT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {OUTPUT_DIR}")
    return checkpoints[-1]


def load_model(checkpoint: Path):
    # device_map={"": 0} (integer) forces all layers onto GPU 0 and bypasses
    # infer_auto_device_map, which would otherwise spill layers to CPU and trigger
    # an accelerate/bitsandbytes incompatibility (Params4bit.__new__ rejects
    # _is_hf_initialized from accelerate's alignment hooks).
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(checkpoint),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def chat(model, tokenizer, system: str | None = None, temperature: float = 0.0) -> None:
    stop_flag = _StopFlag()
    current_system = system

    def _fresh_history():
        h: list[dict] = []
        if current_system:
            h.append({"role": "system", "content": [{"type": "text", "text": current_system}]})
        return h

    history = _fresh_history()

    print('Chat started. Commands: /system "...", /clear, exit (or Ctrl-C). Press B during generation to stop.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if user_input.lower() in ("exit", "quit"):
            print("Bye.")
            break

        if user_input.lower() == "/clear":
            current_system = None
            history = _fresh_history()
            print("[history and system prompt cleared]\n")
            continue

        if user_input.lower().startswith("/system "):
            raw = user_input[8:].strip().strip('"').strip("'")
            current_system = raw if raw else None
            history = _fresh_history()
            label = f'"{current_system}"' if current_system else "None"
            print(f"[system prompt → {label}]\n")
            continue

        if not user_input:
            continue

        history.append({"role": "user", "content": [{"type": "text", "text": user_input}]})

        inputs = tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        stop_flag.clear()
        interrupted = False

        def _listen():
            nonlocal interrupted
            while not stop_flag.is_set():
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key.lower() == b"b":
                        stop_flag.set()
                        interrupted = True
                        print("\n[stopped]", flush=True)
                        break
                time.sleep(0.05)

        listener = threading.Thread(target=_listen, daemon=True)
        listener.start()

        print("\nModel: ", end="", flush=True)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        outputs = model.generate(
            **inputs,
            max_new_tokens=384,
            do_sample=True,
            temperature=1.6,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            use_cache=True,
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([_KeyStopCriteria(stop_flag)]),
        )
        stop_flag.set()  # arrête le listener si génération terminée naturellement

        if not interrupted:
            reply = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )
            history.append({"role": "model", "content": [{"type": "text", "text": reply}]})
        print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Chat with the fine-tuned Gemma checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: latest)")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default), >0 = sampling")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else _last_checkpoint()
    print(f"Loading checkpoint: {checkpoint.name}")
    model, tokenizer = load_model(checkpoint)
    chat(model, tokenizer, system=args.system, temperature=args.temperature)


if __name__ == "__main__":
    main()
