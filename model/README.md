# ParaSpeechCaps Model

This folder contains code related to our ParaSpeechCaps-finetuned Parler-TTS models.

## Inference
We provide three ways to infer with our models:

1. Jupyter Notebook (`inference/run_inference.ipynb`)
2. Command Line Interface (`inference/run_inference.py`)
3. Gradio Web Interface (`inference/run_inference_gradio_app.py`)

For the Gradio web interface, also install Gradio with `pip install gradio`.

The Jupyter notebook shows 3 ways to use our models:
- Basic inference
- Inference with classifier-free guidance
- Inference with ASR-based resampling and classifier-free guidance

The command line interface uses the ASR-based resampling and classifier-free guidance and can be run with:
```bash
python run_inference.py \
    --description "In a clear environment, a male voice speaks with a sad tone." \
    --text "Was that your landlord?" \
    --output_file output.wav
```

The Gradio web interface is a UI version of the command line interface. It can be launched with:
```bash
python run_inference_gradio_app.py
```

### Available Models

We provide two variants of our model:
- [`ajd12342/parler-tts-mini-v1-paraspeechcaps`](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps): Trained on the complete ParaSpeechCaps dataset
- [`ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base`](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base): Trained only on the human-annotated ParaSpeechCaps-Base subset.
