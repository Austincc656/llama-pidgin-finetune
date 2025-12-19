"""
Pidgin Healthcare Assistant - Live Demo
A culturally-adapted AI assistant for healthcare information in Nigerian Pidgin English
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import json

# ============================================
# CONFIGURATION
# ============================================

# Change this to your fine-tuned model path after training
MODEL_PATH = "./pidgin-health-model"
BASE_MODEL = "meta-llama/Llama-3.2-1B"

# For demo without fine-tuned model, set USE_BASE_MODEL = True
USE_BASE_MODEL = False

# ============================================
# MODEL LOADING
# ============================================

model = None
tokenizer = None


def load_model():
    """Load the model and tokenizer."""
    global model, tokenizer

    print("Loading model...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    if USE_BASE_MODEL:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        if USE_BASE_MODEL:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    else:
        if USE_BASE_MODEL:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base_model, MODEL_PATH)

    model.eval()
    print("Model loaded successfully!")


def generate_response(question, max_tokens=256, temperature=0.7):
    """Generate a response to a healthcare question in Pidgin."""
    if model is None:
        return "Model no don load yet. Abeg wait small..."

    prompt = f"""### Question:
{question}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the answer part
    if "### Answer:" in response:
        response = response.split("### Answer:")[-1].strip()

    return response


# ============================================
# HOMEPAGE HTML
# ============================================

HOMEPAGE_HTML = """
<div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #2E7D32; font-size: 2.5em; margin-bottom: 10px;">
        Pidgin Healthcare Assistant
    </h1>
    <p style="font-size: 1.3em; color: #1565C0; margin-bottom: 20px;">
        AI Wey Sabi Helep You With Health Mata For Pidgin
    </p>

    <div style="background: linear-gradient(135deg, #E8F5E9 0%, #E3F2FD 100%);
                padding: 25px; border-radius: 15px; margin: 20px 0;">
        <h2 style="color: #1B5E20;">Wetin Be This?</h2>
        <p style="font-size: 1.1em; line-height: 1.6; color: #333;">
            This na AI assistant wey dem train to answer health questions for
            <strong>Nigerian Pidgin English</strong>. E fit help you understand
            health information for language wey you sabi well-well.
        </p>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px; margin: 25px 0;">
        <div style="background: #FFF3E0; padding: 20px; border-radius: 10px;">
            <h3 style="color: #E65100;">ARV & HIV</h3>
            <p>Information about ARV drugs and HIV treatment</p>
        </div>
        <div style="background: #E8EAF6; padding: 20px; border-radius: 10px;">
            <h3 style="color: #283593;">Side Effects</h3>
            <p>Learn about medication side effects</p>
        </div>
        <div style="background: #FCE4EC; padding: 20px; border-radius: 10px;">
            <h3 style="color: #AD1457;">Pregnancy</h3>
            <p>Health info for pregnant women</p>
        </div>
        <div style="background: #E0F7FA; padding: 20px; border-radius: 10px;">
            <h3 style="color: #00695C;">General Health</h3>
            <p>Everyday health questions</p>
        </div>
    </div>

    <div style="background: #FFF8E1; padding: 20px; border-radius: 10px;
                border-left: 4px solid #FFA000; margin: 20px 0;">
        <p style="margin: 0; color: #5D4037;">
            <strong>Important:</strong> This AI na for information purposes only.
            E no fit replace advice from real doctor or healthcare provider.
            If you get serious health wahala, abeg go see doctor!
        </p>
    </div>

    <div style="margin-top: 20px; padding: 15px; background: #ECEFF1; border-radius: 10px;">
        <p style="color: #455A64; margin: 0;">
            <strong>How to Use:</strong> Just type your health question for Pidgin
            or English for the box below, then click "Ask Question" button.
        </p>
    </div>
</div>
"""

# ============================================
# EXAMPLE QUESTIONS
# ============================================

EXAMPLES = [
    ["Wetin be ARV drugs?"],
    ["How ARV drugs dey work for body?"],
    ["E dey safe make I drink alcohol while I dey take ARV?"],
    ["Wetin go happen if I miss one dose of my medication?"],
    ["I fit born pikin while I dey on ARV treatment?"],
    ["How I go know say my treatment dey work?"],
]

# ============================================
# GRADIO INTERFACE
# ============================================


def chat_response(message, history, max_tokens, temperature):
    """Handle chat messages."""
    response = generate_response(message, int(max_tokens), temperature)
    return response


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="Pidgin Healthcare Assistant",
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
        css="""
        .gradio-container { max-width: 900px !important; }
        footer { display: none !important; }
        """
    ) as demo:

        # Homepage
        gr.HTML(HOMEPAGE_HTML)

        gr.Markdown("---")

        # Chat Interface
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Chat History",
                    height=400,
                    show_label=True,
                )
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Type your health question here... (e.g., 'Wetin be ARV drugs?')",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Ask Question", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear Chat", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=256,
                    step=10,
                    label="Max Response Length",
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Creativity",
                )

        # Examples
        gr.Markdown("### Try These Questions:")
        gr.Examples(
            examples=EXAMPLES,
            inputs=msg,
            label="",
        )

        # Event handlers
        def respond(message, chat_history, max_tok, temp):
            if not message.strip():
                return "", chat_history
            bot_message = chat_response(message, chat_history, max_tok, temp)
            chat_history.append((message, bot_message))
            return "", chat_history

        submit_btn.click(
            respond,
            [msg, chatbot, max_tokens, temperature],
            [msg, chatbot],
        )
        msg.submit(
            respond,
            [msg, chatbot, max_tokens, temperature],
            [msg, chatbot],
        )
        clear_btn.click(lambda: (None, []), None, [msg, chatbot])

        # Footer
        gr.Markdown("""
        ---
        <center>
        <p style="color: #666;">
        Built with Gradio | Fine-tuned on Nigerian Pidgin Healthcare FAQs<br>
        <small>This is a research project - Always consult a healthcare professional</small>
        </p>
        </center>
        """)

    return demo


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 50)
    print("Pidgin Healthcare Assistant - Starting...")
    print("=" * 50)

    # Load model
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        print("Demo will run but responses won't work until model is trained.")

    # Create and launch demo
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates a public link
        show_error=True,
    )
