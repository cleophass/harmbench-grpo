import unsloth  # must be first before trl/transformers/peft

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from judge import score_eval_batch
from logger import RunLogger
from model import GemmaModel

load_dotenv()
console = Console()

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "harmbench_behaviors_text_all.csv"
LOG_DIR = ROOT / "outputs" / "logs"
DEFAULT_N_PROMPTS = 100


def compute_asr(model: GemmaModel | None = None, n_prompts: int = DEFAULT_N_PROMPTS, label: str = "eval") -> float:
    df = pd.read_csv(DATA_PATH)
    df = df[df["SemanticCategory"] != "copyright"].reset_index(drop=True)
    prompts = df["Behavior"].tolist()[:n_prompts]

    if model is None:
        model = GemmaModel().for_inference()

    # Phase 1: génère toutes les réponses
    console.print(f"\n[bold cyan]► {label} — generating {n_prompts} responses[/bold cyan]")
    responses: list[str] = []
    for prompt in tqdm(prompts, desc="generate", unit="prompt"):
        responses.append(model.generate(prompt))

    # Phase 2: juge tout en batch (comparatif)
    console.print(f"\n[bold cyan]► judging batch[/bold cyan]")
    scores = score_eval_batch(list(zip(prompts, responses)))

    # Phase 3: log + ASR final
    logger = RunLogger(total=n_prompts, label=label, log_dir=LOG_DIR)
    for prompt, response, score in zip(prompts, responses, scores):
        logger.log(prompt, response, score)
    return logger.close()


if __name__ == "__main__":
    compute_asr()
