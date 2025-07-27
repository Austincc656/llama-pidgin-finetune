# llama-pidgin-finetune
Fine-tuning LLaMA on Pidgin health data using Unsloth
# 🧠 LLaMA Fine-Tuning for Pidgin Healthcare Communication 🇳🇬

This project fine-tunes a LLaMA model using the [Unsloth](https://github.com/unslothai/unsloth) library on a curated dataset of healthcare FAQs translated into **Nigerian Pidgin English**.

It supports ongoing research on improving access to culturally relevant health information for low-resource languages.

---

## 🚀 Features

- LoRA-based fine-tuning with Unsloth (efficient and fast)
- Healthcare Q&A dataset in Pidgin English
- Preprocessing, training, and evaluation all in Google Colab
- Uses Hugging Face `datasets` and `transformers`

---

## 📁 Project Structure

"# llama-pidgin-finetune" 


## 🧠 LoRA-Based Fine-Tuning Configuration (Unsloth)

This project uses **parameter-efficient fine-tuning (PEFT)** with **LoRA (Low-Rank Adaptation)** via the [Unsloth](https://github.com/unslothai/unsloth) framework, making it possible to fine-tune the LLaMA 3 model efficiently on limited hardware (e.g. Google Colab with T4 GPU).

### ✅ LoRA Configuration Code

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.0,  # Use 0.05 if stronger regularization is desired
    bias = "none"
)

