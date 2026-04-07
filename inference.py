import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from prompts import SYSTEM_PROMPT


def _pick_device(device: str) -> str:
    device = (device or "auto").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(base_model_id: str, lora_path: str = None, device: str = "auto"):
    device = _pick_device(device)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    if lora_path and lora_path.strip() and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(model, lora_path)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def build_prompt(user_text: str) -> str:
    return f"""<|system|>
{SYSTEM_PROMPT}
<|user|>
{user_text}
<|assistant|>
"""


def generate_reply(
    tokenizer,
    model,
    user_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    if not user_text or not str(user_text).strip():
        return "Abeg type your question make I help you."

    prompt = build_prompt(str(user_text).strip())

    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if "<|assistant|>" in decoded:
        reply = decoded.split("<|assistant|>", 1)[-1].strip()
    else:
        reply = decoded.strip()

    return reply or "I no sure say I get correct answer for this one. You fit rephrase am?"