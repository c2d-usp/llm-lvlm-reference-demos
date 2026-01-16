# References for LLMs and LVLMs Demonstrations

This repository contains reference scripts for demonstrating and comparing Large Language Models (LLMs) and Multimodal Models (LLaVA). We use Gradio to build these simple interactive interfaces.

## File Structure

We organize our repository as follows.

- `llms/`: Contains scripts for comparing two LLMs and for interacting with a single LLM. Both demonstrations follow a chat format.
- `llava/`: Provides scripts for comparing two LLaVA-based models and for interacting with a single one. Both demonstrations follow a format including text and image input.
- `requirements.txt`: Requirements for running the scripts. We recommend creating a virtual environment. For installation, use the command: `pip install -r requirements.txt`.

## About the Scripts

To allow quick and easy customization, we isolate key variables of the demonstrations at the top of each script. For example, in the `gradio_llava_two_devices.py` script, we present three categories of variables:
### Model Configurations
- `MODEL_ID_1`: HuggingFace ID of the first model
- `PROCESSOR_ID_1`: HuggingFace ID to load the first processor
- `MODEL_ID_2`: HuggingFace ID of the second model
- `PROCESSOR_ID_2`: HuggingFace ID to load the second processor
- `DEVICE_1`: Device to load the first model
- `DEVICE_2`: Device to load the second model

### Gradio Interface Configurations
- `DEMO_TITLE`: The title of the demonstration
- `DEMO_DESCRIPTION_MD`: Markdown description of the demonstration shown at the top of the interface
- `MODEL_1_TITLE_MD`: Title for the first model's column
- `MODEL_2_TITLE_MD`: Title for the second model's column

### Generation Parameters
- `MAX_NEW_TOKENS`: Maximum number of tokens to generate
- `TEMPERATURE`: Adjusts the probability distribution of the next possible tokens. High temperature flattens the distribution, increasing the probability of low-probability tokens. Low temperature maintains the most likely tokens dominant.
- `TOP_P`: Controls which tokens are available to sample. For example, TOP_P = 0.8 enables sampling only the top 80% most probable tokens.
- `SAMPLE`: Enable or disable sampling during generation