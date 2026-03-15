import gradio as gr
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# --- Model Configurations ---
MODEL_ID_1 = "c2d-usp/layer_wise_lora_Qwen3-VL_8B_chartqa_1epoch"
PROCESSOR_ID_1 = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE_1 = "cuda:0"

# -- Gradio Interface Configurations ---
DEMO_TITLE = "Qwen3-VL Demo" # Note: Appears in the browser tab
DEMO_DESCRIPTION_MD = "# Demonstration "
MODEL_1_TITLE_MD = "### Finetuned (50%) Qwen3-VL-8B-Instruct - ChartQA"

# --- Generation Parameters ---
MAX_NEW_TOKENS = 5000
TEMPERATURE = 0.7
TOP_P = 0.8
SAMPLE = True



processor1 = AutoProcessor.from_pretrained(PROCESSOR_ID_1)
model1 = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_1, 
    torch_dtype=torch.float16
).to(DEVICE_1)

def run_inference(image, text, model, processor, device):
    """
    Auxiliary function that processes the input and generates the response
    for a specific model and device.
    """
    if image is None:
        return "Upload an image!"
    
    if not text:
        text = "Describe this image."

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"} if image is not None else None,
                {"type": "text", "text": text}
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
 
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)


    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=SAMPLE, 
            temperature=TEMPERATURE,
            top_p=TOP_P
        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = generate_ids[:, input_length:]

    output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return output


def generate_device_1(image, text):
    try:
        return run_inference(image, text, model1, processor1, DEVICE_1)
    except Exception as e:
        return f"Device 1 Error: {str(e)}"

css = """
#main {
    max-width: 750px;
    margin: auto;
}
"""

with gr.Blocks(title=DEMO_TITLE,css = css) as demo:
    gr.Markdown(DEMO_DESCRIPTION_MD)

    with gr.Row():
        with gr.Column(variant="panel",elem_id="main"):
            gr.Markdown(MODEL_1_TITLE_MD)
            
            img_input_1 = gr.Image(type="pil", label="Input Image")
            txt_input_1 = gr.Textbox(lines=2, placeholder="Ask something...", label="Prompt")
            btn_1 = gr.Button("Generate", variant="primary")
            output_1 = gr.Textbox(label="Model Output",lines=8)
            
            btn_1.click(fn=generate_device_1, inputs=[img_input_1, txt_input_1], outputs=output_1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, theme=gr.themes.Base())