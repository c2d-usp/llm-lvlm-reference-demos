import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Model Configurations ---
MODEL_ID_1 = "meta-llama/Llama-2-7b-hf"
TOKENIZER_ID_1 = "meta-llama/Llama-2-7b-hf"
MODEL_ID_2 = "c2d-usp/MoP_llama_2_20_percent_compression_2_epochs" 
TOKENIZER_ID_2 = "meta-llama/Llama-2-7b-hf"
DEVICE_1 = "cuda:0"
DEVICE_2 = "cuda:1"

# -- Gradio Interface Configurations ---
DEMO_TITLE = "LLaMA Multi-GPU Chat Comparison"
DEMO_DESCRIPTION_MD = "# Comparison - Dense Model x Pruned Model (80% Parameters)"
MODEL_1_TITLE_MD = "### Dense Model (6.74B)"
MODEL_2_TITLE_MD = "### Pruned Model (5.38B)"

# --- Generation Parameters ---
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.6
TOP_P = 0.8
SAMPLE = True


tokenizer1 = AutoTokenizer.from_pretrained(TOKENIZER_ID_1)
model1 = AutoModelForCausalLM.from_pretrained(MODEL_ID_1, torch_dtype=torch.float16).to(DEVICE_1)

tokenizer2 = AutoTokenizer.from_pretrained(TOKENIZER_ID_2)
model2 = AutoModelForCausalLM.from_pretrained(MODEL_ID_2, torch_dtype=torch.float16).to(DEVICE_2)

def run_inference(text, model, tokenizer, device):
    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n" + text +
        "\n\n### Response:\n"
    )
    inputs = tokenizer(text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=SAMPLE, 
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
    output = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    return output.split("### Response:")[-1].strip()

def chat_model_1(message, history):
    history.append({"role": "user", "content": message})
    try:
        response = run_inference(message, model1, tokenizer1, DEVICE_1)
    except Exception as e:
        response = f"Device 1 Error: {str(e)}"
    history.append({"role": "assistant", "content": response})
    
    return "", history

def chat_model_2(message, history):
    history.append({"role": "user", "content": message})
    try:
        response = run_inference(message, model2, tokenizer2, DEVICE_2)
    except Exception as e:
        response = f"Device 2 Error: {str(e)}"
    history.append({"role": "assistant", "content": response})
    return "", history

with gr.Blocks(title=DEMO_TITLE) as demo:
    gr.Markdown(DEMO_DESCRIPTION_MD)

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown(MODEL_1_TITLE_MD)
            chatbot1 = gr.Chatbot(label="Chat 1", height=450)
            msg1 = gr.Textbox(placeholder="Ask Something", label="Input")
            with gr.Row():
                btn1 = gr.Button("Send", variant="primary")
                clear1 = gr.ClearButton([msg1, chatbot1])

        with gr.Column(variant="panel"):
            gr.Markdown(MODEL_2_TITLE_MD)
            chatbot2 = gr.Chatbot(label="Chat 2", height=450)
            msg2 = gr.Textbox(placeholder="Ask Something", label="Input")
            with gr.Row():
                btn2 = gr.Button("Send", variant="primary")
                clear2 = gr.ClearButton([msg2, chatbot2])

    btn1.click(chat_model_1, inputs=[msg1, chatbot1], outputs=[msg1, chatbot1])
    msg1.submit(chat_model_1, inputs=[msg1, chatbot1], outputs=[msg1, chatbot1])

    btn2.click(chat_model_2, inputs=[msg2, chatbot2], outputs=[msg2, chatbot2])
    msg2.submit(chat_model_2, inputs=[msg2, chatbot2], outputs=[msg2, chatbot2])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, theme=gr.themes.Base())