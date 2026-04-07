# 🇳🇬 AYO – Pidgin Health Language Assistant

> Fine-tuned Large Language Model (LLM) for culturally relevant healthcare communication in Nigerian Pidgin English.

---

## 🚀 Project Overview

AYO (A Pidgin Health Language Assistant) is an AI-powered chatbot designed to improve access to healthcare information for Nigerian Pidgin English speakers.

This project was developed as part of a Master's thesis in Business Information Systems, focusing on adapting Large Language Models (LLMs) for low-resource languages in the public health domain.

The system combines:
- Fine-tuned LLMs (LLaMA-based)
- Domain-specific healthcare FAQ dataset
- Prompt alignment for cultural and linguistic relevance
- Interactive chatbot interface using Gradio

---

## 🧠 Key Features

- 💬 Conversational AI chatbot interface
- 🌍 Nigerian Pidgin English responses
- 🏥 Healthcare-focused knowledge base
- ⚙️ LoRA fine-tuning for efficient model adaptation
- 🎯 Culturally aligned responses for better accessibility
- 🖥️ Interactive UI built with Gradio
- 🔒 Safety-aware responses (no diagnosis / emergency guidance)

---

## 🏗️ System Architecture

User Input  
↓  
Prompt Engineering (Pidgin-aligned system prompt)  
↓  
Fine-tuned LLM (LoRA adapter on base model)  
↓  
Response Generation (controlled decoding)  
↓  
Chat Interface (Gradio UI)

---

## 📊 Dataset

The model is trained on a curated healthcare FAQ dataset translated into Nigerian Pidgin English.

Data pipeline:
- Excel → JSON conversion
- Cleaning & structuring
- Hugging Face dataset format

---

## ⚙️ Tech Stack

- Python
- Hugging Face Transformers
- LoRA (PEFT)
- Unsloth (efficient fine-tuning)
- Gradio (UI)
- PyTorch

---

## 🧪 Model Setup

```env
BASE_MODEL_ID=unsloth/llama-3-8b-bnb-4bit
LORA_PATH=./models/ayo-lora
DEVICE=cpu
