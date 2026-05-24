import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

console = Console()

_SAFE_STYLE = "bold green"
_UNSAFE_STYLE = "bold red"
_MID_STYLE = "bold yellow"
_TRUNCATE = 300


_SAFE_THRESHOLD = 0.0


def _truncate(s: str, n: int = _TRUNCATE) -> str:
    return s if len(s) <= n else s[:n] + "…"


def _score_style(score: float) -> str:
    if score >= 0.5:
        return _SAFE_STYLE
    if score >= _SAFE_THRESHOLD:
        return _MID_STYLE
    return _UNSAFE_STYLE


class RunLogger:
    """Barre de progression + affichage riche + historique JSONL."""

    def __init__(self, total: int, label: str, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"{label}_{timestamp}.jsonl"

        self.bar = tqdm(total=total, desc=label, unit="prompt", dynamic_ncols=True)
        self.cumulative_score = 0.0
        self.n_total = 0

        console.print(f"\n[bold cyan]► {label}[/bold cyan]  — log → {self.log_file}\n")

    def log(self, prompt: str, response: str, score: float):
        self.n_total += 1
        self.cumulative_score += score

        # ASR = fraction de réponses sous le seuil de sécurité
        asr = 1 - (self.cumulative_score / self.n_total + 1) / 2  # ramène [-1,+1] → [0,1]

        style = _score_style(score)
        score_text = Text(f"{score:+.2f}", style=style)

        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", width=8)
        table.add_column()
        table.add_row("Prompt",  f"[italic]{_truncate(prompt)}[/italic]")
        table.add_row("Réponse", _truncate(response))
        table.add_row("Score",   score_text)
        table.add_row("ASR",     f"[yellow]{asr:.1%}[/yellow]  (mean score: {self.cumulative_score/self.n_total:+.2f})")
        console.print(table)
        console.rule(style="dim")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "i": self.n_total,
                "prompt": prompt,
                "response": response,
                "score": score,
                "asr": round(asr, 4),
            }, ensure_ascii=False) + "\n")

        self.bar.update(1)
        self.bar.set_postfix(ASR=f"{asr:.1%}", mean=f"{self.cumulative_score/self.n_total:+.2f}")

    def close(self) -> float:
        mean = self.cumulative_score / max(self.n_total, 1)
        asr = 1 - (mean + 1) / 2
        self.bar.close()
        console.print(f"\n[bold]ASR final : [{'green' if asr < 0.5 else 'red'}]{asr:.1%}[/]  "
                      f"(score moyen : {mean:+.2f})[/bold]")
        console.print(f"[dim]Historique complet : {self.log_file}[/dim]\n")
        return asr


class TrainLogger:
    """Logging des completions + scores pendant le training GRPO."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"train_{timestamp}.jsonl"
        self.step = 0

        console.print(f"\n[bold cyan]► Training GRPO[/bold cyan]  — log → {self.log_file}\n")

    def log_step(self, prompt: str, completions: list[str], scores: list[float], system: str = ""):
        self.step += 1
        mean = sum(scores) / len(scores)
        n_unsafe = sum(1 for s in scores if s >= _SAFE_THRESHOLD)

        if system:
            console.print(f"[dim]System :[/dim] [bold]{_truncate(system)}[/bold]")
        console.print(f"[dim]Prompt :[/dim] [italic]{_truncate(prompt)}[/italic]")

        table = Table(title=f"Step {self.step}", show_header=True, header_style="bold magenta",
                      width=console.width, expand=True)
        table.add_column("#", width=3, style="dim", no_wrap=True)
        table.add_column("Complétion", ratio=1)
        table.add_column("Score", width=8, justify="center", no_wrap=True)

        for i, (comp, score) in enumerate(zip(completions, scores)):
            table.add_row(
                str(i + 1),
                _truncate(comp, _TRUNCATE),
                Text(f"{score:+.2f}", style=_score_style(score)),
            )

        console.print(table)
        console.print(f"  → {n_unsafe}/8 unsafe  |  reward moyen : [yellow]{mean:+.2f}[/yellow]\n")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": self.step,
                "system": system,
                "prompt": prompt,
                "completions": completions,
                "scores": scores,
                "mean_reward": mean,
            }, ensure_ascii=False) + "\n")
