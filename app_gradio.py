import os
import gradio as gr
from dotenv import load_dotenv

from inference import load_model, generate_reply
from prompts import INTRO_TEXT

load_dotenv()

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "").strip()
LORA_PATH = os.getenv("LORA_PATH", "").strip()
DEVICE = os.getenv("DEVICE", "auto").strip()

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

nigeria_theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="green",
    neutral_hue="gray",
).set(
    body_background_fill="#FFFFFF",
    block_background_fill="#FFFFFF",
    block_border_color="#008753",
    input_background_fill="#FFFFFF",
    input_border_color="#008753",
    button_primary_background_fill="#008753",
    button_primary_background_fill_hover="#006b42",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#E6F4EE",
    button_secondary_text_color="#008753",
)

tokenizer = None
model = None


def init_once():
    global tokenizer, model
    if tokenizer is None or model is None:
        if not BASE_MODEL_ID:
            raise ValueError("BASE_MODEL_ID is missing. Add it to your .env file.")
        tokenizer_, model_, _ = load_model(
            base_model_id=BASE_MODEL_ID,
            lora_path=LORA_PATH if LORA_PATH else None,
            device=DEVICE,
        )
        tokenizer, model = tokenizer_, model_


def chat_fn(message, history):
    try:
        init_once()
        return generate_reply(
            tokenizer,
            model,
            message,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    except Exception as e:
        return f"Error: {type(e).__name__} – {e}"


with gr.Blocks(
    title="AYO – A Pidgin Health Language Assistant",
    theme=nigeria_theme,
) as demo:
    gr.Markdown(INTRO_TEXT)
    gr.Markdown("<hr style='border:1px solid #008753'>")

    gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(height=420),
        textbox=gr.Textbox(
            placeholder="Type your health question for Pidgin…",
            container=False,
        ),
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)