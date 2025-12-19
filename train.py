"""
Pidgin Healthcare LLaMA Fine-Tuning Script
Fine-tunes LLaMA 3 on Nigerian Pidgin English healthcare FAQs using LoRA
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================================
# CONFIGURATION
# ============================================

MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Use smaller model for demo, change to 3B/8B if you have GPU
OUTPUT_DIR = "./pidgin-health-model"
DATA_FILE = "faq_pidgin.json"

# LoRA Configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training Configuration
EPOCHS = 3
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MAX_LENGTH = 512


def load_data(file_path):
    """Load and format the Pidgin healthcare FAQ data."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format as instruction-response pairs
    formatted_data = []
    for item in data:
        text = f"""### Question:
{item['question']}

### Answer:
{item['answer']}"""
        formatted_data.append({"text": text})

    return Dataset.from_list(formatted_data)


def tokenize_data(dataset, tokenizer, max_length):
    """Tokenize the dataset."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return dataset.map(tokenize_function, batched=True, remove_columns=["text"])


def main():
    print("=" * 50)
    print("Pidgin Healthcare LLaMA Fine-Tuning")
    print("=" * 50)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("WARNING: Training on CPU will be very slow. GPU recommended.")

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model with quantization for memory efficiency
    print("\n[2/5] Loading model...")

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

    # Apply LoRA
    print("\n[3/5] Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare data
    print("\n[4/5] Loading and tokenizing data...")
    dataset = load_data(DATA_FILE)
    print(f"Loaded {len(dataset)} examples")

    tokenized_dataset = tokenize_data(dataset, tokenizer, MAX_LENGTH)

    # Split into train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        fp16=device == "cuda",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Initialize trainer
    print("\n[5/5] Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    # Save the model
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
