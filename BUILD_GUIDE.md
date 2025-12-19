# Live Demo Build Guide

Build and run the Pidgin Healthcare Assistant demo in 3 steps.

---

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU (recommended):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

### Step 2: Train the Model (Optional)

If you want to fine-tune the model on the Pidgin healthcare data:

```bash
python train.py
```

This will:
- Load LLaMA 3.2-1B base model
- Apply LoRA fine-tuning on `faq_pidgin.json`
- Save the trained model to `./pidgin-health-model/`

**Training time:**
- GPU (T4/V100): ~30-60 minutes
- CPU: Several hours (not recommended)

**Skip training?** The demo can run with just the base model. Edit `app.py` and set:
```python
USE_BASE_MODEL = True
```

---

### Step 3: Run the Demo

```bash
python app.py
```

The demo will start at:
- **Local:** http://localhost:7860
- **Public:** A shareable link will be printed (if `share=True`)

---

## What You'll See

The demo includes:

1. **Homepage** explaining the app in Pidgin English
2. **Chat interface** for asking health questions
3. **Example questions** to try
4. **Settings** for response length and creativity

---

## File Structure

```
llama-pidgin-finetune/
├── app.py              # Gradio demo app
├── train.py            # Training script
├── requirements.txt    # Python dependencies
├── faq_pidgin.json     # Healthcare Q&A dataset
├── BUILD_GUIDE.md      # This file
└── pidgin-health-model/ # (created after training)
```

---

## Google Colab Setup

For free GPU access, use Google Colab:

```python
# Cell 1: Install dependencies
!pip install torch transformers accelerate peft bitsandbytes datasets gradio

# Cell 2: Clone your repo
!git clone https://github.com/YOUR_USERNAME/llama-pidgin-finetune.git
%cd llama-pidgin-finetune

# Cell 3: Train (optional)
!python train.py

# Cell 4: Run demo
!python app.py
```

---

## Hugging Face Spaces Deployment

To deploy on Hugging Face Spaces:

1. Create a new Space at huggingface.co/spaces
2. Select "Gradio" as the SDK
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   - `faq_pidgin.json`
   - Your trained model folder (or use base model)

4. The Space will auto-build and deploy

---

## Configuration Options

In `app.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./pidgin-health-model` | Path to fine-tuned model |
| `BASE_MODEL` | `meta-llama/Llama-3.2-1B` | Base LLaMA model |
| `USE_BASE_MODEL` | `False` | Use base model without fine-tuning |

In `train.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | `3` | Number of training epochs |
| `BATCH_SIZE` | `4` | Training batch size |
| `LORA_R` | `8` | LoRA rank |
| `MAX_LENGTH` | `512` | Maximum sequence length |

---

## Troubleshooting

**"CUDA out of memory"**
- Reduce `BATCH_SIZE` in `train.py`
- Use a smaller model

**"Model not found"**
- Run `train.py` first, or set `USE_BASE_MODEL = True`

**Slow generation**
- Use GPU for faster inference
- Reduce `max_tokens` in the demo

---

## Example Questions to Try

- "Wetin be ARV drugs?"
- "How ARV drugs dey work?"
- "E dey safe make I drink alcohol while I dey take ARV?"
- "I fit born pikin while I dey on treatment?"
- "Wetin go happen if I miss my dose?"
