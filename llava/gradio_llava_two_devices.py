import gradio as gr
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- Model Configurations ---
MODEL_ID_1 = "llava-hf/llava-1.5-7b-hf"
PROCESSOR_ID_1 = "llava-hf/llava-1.5-7b-hf"
MODEL_ID_2 = "c2d-usp/MoP_llava_1.5_20_percent_compression_2_epochs" 
PROCESSOR_ID_2 = "llava-hf/llava-1.5-7b-hf"
DEVICE_1 = "cuda:0"
DEVICE_2 = "cuda:1"

# -- Gradio Interface Configurations ---
DEMO_TITLE = "LLaVA Multi-GPU Comparison"
DEMO_DESCRIPTION_MD = "# Comparison - Dense Model x Pruned Model (80% Parameters)"
MODEL_1_TITLE_MD = "### Dense Model (Parameters = 7.06B)"
MODEL_2_TITLE_MD = "### Pruned Model (Parameters = 5.71B)"

# --- Generation Parameters ---
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.5
SAMPLE = True



processor1 = AutoProcessor.from_pretrained(PROCESSOR_ID_1)
model1 = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID_1, 
    torch_dtype=torch.float16
).to(DEVICE_1)

processor2 = AutoProcessor.from_pretrained(PROCESSOR_ID_2)
model2 = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID_2, 
    torch_dtype=torch.float16
).to(DEVICE_2)

def run_inference(image, text, model, processor, device):
    """
    Auxiliary function that processes the input and generates the response
    for a specific model and device.
    """
    if image is None:
        return "Upload an image!"
    
    if not text:
        text = "Describe this image."

    prompt = f"USER: <image>\n{text}\nASSISTANT:"

 
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)


    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs, 
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=SAMPLE, 
            temperature=TEMPERATURE,
            top_p=TOP_P
        )


    output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if "ASSISTANT:" in output:
        return output.split("ASSISTANT:")[-1].strip()
    
    return output


def generate_device_1(image, text):
    try:
        return run_inference(image, text, model1, processor1, DEVICE_1)
    except Exception as e:
        return f"Device 1 Error: {str(e)}"

def generate_device_2(image, text):
    try:
        return run_inference(image, text, model2, processor2, DEVICE_2)
    except Exception as e:
        return f"Device 2 Error: {str(e)}"

with gr.Blocks(title=DEMO_TITLE) as demo:
    gr.Markdown(DEMO_DESCRIPTION_MD)

    with gr.Row():
        with gr.Column(variant="panel"):
            gr.Markdown(MODEL_1_TITLE_MD)
            
            img_input_1 = gr.Image(type="pil", label="Input Image")
            txt_input_1 = gr.Textbox(lines=2, placeholder="Ask something...", label="Prompt")
            btn_1 = gr.Button("Generate", variant="primary")
            output_1 = gr.Textbox(label="Model Output",lines=8)
            
            btn_1.click(fn=generate_device_1, inputs=[img_input_1, txt_input_1], outputs=output_1)

        with gr.Column(variant="panel"):
            gr.Markdown(MODEL_2_TITLE_MD)
            
            img_input_2 = gr.Image(type="pil", label="Input Image")
            txt_input_2 = gr.Textbox(lines=2, placeholder="Ask something...", label="Prompt")
            btn_2 = gr.Button("Generate", variant="primary")
            output_2 = gr.Textbox(label="Model Output",lines=8)

            btn_2.click(fn=generate_device_2, inputs=[img_input_2, txt_input_2], outputs=output_2)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False, theme=gr.themes.Base())