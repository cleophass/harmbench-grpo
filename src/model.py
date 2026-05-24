from unsloth import FastLanguageModel

MODEL_ID = "google/gemma-4-E4B-it"
MAX_SEQ_LENGTH = 2048

LORA_CONFIG = dict(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)


class GemmaModel:
    def __init__(self, model_id: str = MODEL_ID, max_new_tokens: int = 512):
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
            device_map={"": "cuda:0"},  
        )

    def for_inference(self) -> "GemmaModel":
        FastLanguageModel.for_inference(self.model)
        return self

    def for_training(self) -> "GemmaModel":
        self.model = FastLanguageModel.get_peft_model(self.model, **LORA_CONFIG)
        return self

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0]

    def generate_batch(self, prompts: list[str]) -> list[str]:
        messages_batch = [
            [{"role": "user", "content": [{"type": "text", "text": p}]}]
            for p in prompts
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            use_cache=True,
        )
        return [
            self.tokenizer.decode(out[inputs.shape[-1]:], skip_special_tokens=True)
            for out in outputs
        ]
