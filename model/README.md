# ParaSpeechCaps Models

This folder contains code related to our ParaSpeechCaps-finetuned Parler-TTS models. We provide two variants of our model:
- [`ajd12342/parler-tts-mini-v1-paraspeechcaps`](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps): Trained on the complete ParaSpeechCaps dataset
- [`ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base`](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base): Trained only on the human-annotated ParaSpeechCaps-Base subset.

Model weights are licensed under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Table of Contents
1. [Installation](#1-installation)
2. [Inference](#2-inference)

## 1. Installation
This repository has been tested with Python 3.11 (`conda create -n paraspeechcaps python=3.11`), but most other versions should probably work.
```bash
git clone https://github.com/ajd12342/paraspeechcaps.git
cd paraspeechcaps/model/parler-tts
pip install -e .[train]
pip install gradio # Only needed if you want to use the Gradio web interface
```

## 2. Inference
We provide three ways to infer with our models:

1. Jupyter Notebook (`inference/run_inference.ipynb`)
2. Command Line Interface (`inference/run_inference.py`)
3. Gradio Web Interface (`inference/run_inference_gradio_app.py`)

The Jupyter notebook shows 3 ways to use our models:
- Basic inference
- Inference with classifier-free guidance
- Inference with ASR-based resampling and classifier-free guidance

The command line interface uses the ASR-based resampling and classifier-free guidance. Example usage:
```bash
python run_inference.py \
    --description "In a clear environment, a male voice speaks with a sad tone." \
    --text "Was that your landlord?" \
    --output_file output.wav
```
Run `python run_inference.py --help` to see all the available inference options.

The Gradio web interface is a UI version of the command line interface. It can be launched with:
```bash
python run_inference_gradio_app.py
```
