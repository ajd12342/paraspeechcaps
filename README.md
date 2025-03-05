# ParaSpeechCaps: Scaling Rich Style-Prompted Text-to-Speech Datasets
[![ParaSpeechCaps Dataset](https://img.shields.io/badge/🤗-ParaSpeechCaps%20Dataset-yellow)](https://huggingface.co/datasets/ajd12342/paraspeechcaps)
[![ParaSpeechCaps Full Model](https://img.shields.io/badge/🤗-ParaSpeechCaps%20Full%20Model-blue)](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps)
[![ParaSpeechCaps Base Model](https://img.shields.io/badge/🤗-ParaSpeechCaps%20Base%20Model-green)](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base)
[![Paper](https://img.shields.io/badge/Paper-PDF-orange)]()

This repository contains the official code for [Scaling Rich Style-Prompted Text-to-Speech Datasets](). We release ParaSpeechCaps (Paralinguistic Speech Captions), a large-scale dataset that annotates speech utterances with rich style captions at [ajd12342/paraspeechcaps](https://huggingface.co/datasets/ajd12342/paraspeechcaps). We also release [Parler-TTS](https://github.com/huggingface/parler-tts) models finetuned on our dataset at [ajd12342/parler-tts-mini-v1-paraspeechcaps](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps) and [ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base).

Please take a look at our [paper]() and our [demo website](https://paraspeechcaps.github.io/) for more information.

**LICENSE:** This code repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The dataset and models are licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## 1. Overview

[ParaSpeechCaps](https://huggingface.co/datasets/ajd12342/paraspeechcaps) is a large-scale dataset that annotates speech utterances with rich style captions. It supports 59 style tags covering styles like pitch, rhythm, emotion, and more, spanning speaker-level intrinsic style tags and utterance-level situational style tags. It consists of a human-annotated subset ParaSpeechCaps-Base and a large automatically-annotated subset ParaSpeechCaps-Scaled. Our novel pipeline combining off-the-shelf text and speech embedders, classifiers and an audio language model allows us to automatically scale rich tag annotations for such a wide variety of style tags for the first time.

We finetune Parler-TTS on our ParaSpeechCaps dataset to create TTS models that can generate speech while controlling for rich styles (pitch, rhythm, clarity, emotion, etc.) with a textual style prompt ('A male speaker's speech is distinguished by a slurred articulation, delivered at a measured pace in a clear environment.').

## 2. ParaSpeechCaps Dataset
The ParaSpeechCaps dataset is available on the Hugging Face Hub at [`ajd12342/paraspeechcaps`](https://huggingface.co/datasets/ajd12342/paraspeechcaps).

### 2.1 Installation
This repository has been tested with Python 3.11 (`conda create -n paraspeechcaps python=3.11`), but most other versions should probably work.
```bash
pip install datasets
```

### 2.2 Quickstart
```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("ajd12342/paraspeechcaps")

# Load specific splits of the dataset
train_scaled = load_dataset("ajd12342/paraspeechcaps", split="train_scaled")
train_base = load_dataset("ajd12342/paraspeechcaps", split="train_base")
dev = load_dataset("ajd12342/paraspeechcaps", split="dev")
holdout = load_dataset("ajd12342/paraspeechcaps", split="holdout")

# View a single example
example = train_base[0]
print(example)
```

### 2.2 Detailed Usage
Please refer to the [dataset](https://github.com/ajd12342/paraspeechcaps/tree/main/dataset) folder for details on how to use it.

### TODOS
- [ ] Release code for our human annotation pipeline
- [ ] Release code for our automatic annotation pipeline

## 3. ParaSpeechCaps Models

The ParaSpeechCaps models are available on the Hugging Face Hub at [`ajd12342/parler-tts-mini-v1-paraspeechcaps`](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps) (trained on the full dataset) and [`ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base`](https://huggingface.co/ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base) (trained on the human-annotated subset).

### 3.1 Installation
This repository has been tested with Python 3.11 (`conda create -n paraspeechcaps python=3.11`), but most other versions should probably work.
```bash
git clone https://github.com/ajd12342/paraspeechcaps.git
cd paraspeechcaps/model/parler-tts
pip install -e .[train]
```

NOTE: We recommend you follow the installation instructions above because our fork of Parler-TTS adds support for inference-time classifier-free guidance (which consistently improves performance) and new training scripts. However, if you only wish to perform model inference and don't want to use classifier-free guidance, our models are fully compatible with the original [Parler-TTS](https://github.com/huggingface/parler-tts/tree/d108732cd57788ec86bc857d99a6cabd66663d68) repository as well.

### 3.2 Quickstart

#### 3.2.1 Inference

```python
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "ajd12342/parler-tts-mini-v1-paraspeechcaps"
guidance_scale = 1.5

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
description_tokenizer = AutoTokenizer.from_pretrained(model_name)
transcription_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

input_description = "In a clear environment, a male voice speaks with a sad tone.".replace('\n', ' ').rstrip()
input_transcription = "Was that your landlord?".replace('\n', ' ').rstrip()

input_description_tokenized = description_tokenizer(input_description, return_tensors="pt").to(model.device)
input_transcription_tokenized = transcription_tokenizer(input_transcription, return_tensors="pt").to(model.device)

generation = model.generate(input_ids=input_description_tokenized.input_ids, prompt_input_ids=input_transcription_tokenized.input_ids, guidance_scale=guidance_scale)

audio_arr = generation.cpu().numpy().squeeze()
sf.write("output.wav", audio_arr, model.config.sampling_rate)
```
Please refer to the [model](https://github.com/ajd12342/paraspeechcaps/tree/main/model) folder for more inference scripts (including a CLI version, a notebook version, and a gradio demo version).

### TODOS
- [ ] Training and evaluation code
- [ ] Annotation UIs for evaluation metrics

## 4. Citation

If you use this repository, the dataset or models, please cite our work as follows:

## 5. Acknowledgements

We thank the authors of [Parler-TTS](https://github.com/huggingface/parler-tts) for their excellent work on the Parler-TTS model.