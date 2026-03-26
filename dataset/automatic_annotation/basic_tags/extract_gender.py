"""Extract gender predictions and add as a column to a HuggingFace dataset.

Uses the audeering/wav2vec2-large-robust-24-ft-age-gender model to predict
gender (male/female) for each utterance and writes the result back as a
`gender` column in the dataset.

Usage:
    python extract_gender.py \
        --dataset_name ./my-dataset-with-acoustic-values \
        --load_from_disk \
        --output_dir ./my-dataset-with-acoustic-values-and-gender

    python extract_gender.py \
        --dataset_name your-username/my-hf-dataset \
        --output_dir ./my-dataset-with-acoustic-values-and-gender

The input dataset must have an `audio_path` column pointing to audio files.
The output dataset adds a `gender` column with values "male" or "female".
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
import librosa
from datasets import load_dataset, load_from_disk


class ModelHead(nn.Module):
    """Classification head."""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Age and gender prediction model based on wav2vec2."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        # logits_gender shape: [batch, 3] — indices: 0=female, 1=male, 2=child
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def make_predict_gender_fn(processor, model, device, audio_path_column_name):
    """Return a batched map function that adds a 'gender' column."""
    def predict_gender(batch):
        genders = []
        for wav_file in batch[audio_path_column_name]:
            data, sr = librosa.load(wav_file, sr=16000)
            data = data.astype(np.float32)
            y = processor(data, sampling_rate=sr)['input_values'][0].reshape(1, -1)
            y = torch.from_numpy(y).to(device)
            with torch.no_grad():
                _, _, logits_gender = model(y)
            probs = logits_gender.detach().cpu().numpy()[0]  # [female, male, child]
            genders.append("male" if probs[1] > probs[0] else "female")
        return {"gender": genders}
    return predict_gender


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract gender predictions and add as a column to a HuggingFace dataset."
    )
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Path to HF dataset on disk, or HuggingFace Hub dataset name')
    parser.add_argument('--load_from_disk', default=False, action='store_true',
                        help='If set, load the dataset from a local directory instead of the Hub')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the output dataset with the gender column added')
    parser.add_argument('--audio_path_column_name', type=str, default='audio_path',
                        help='Column name containing paths to audio files (default: audio_path)')
    parser.add_argument('--model_name', type=str,
                        default='audeering/wav2vec2-large-robust-24-ft-age-gender',
                        help='HuggingFace model name for age/gender prediction')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of examples per batch during inference (default: 16)')
    parser.add_argument('--cpu_num_workers', type=int, default=1,
                        help='Number of workers for dataset.map (default: 1)')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = AgeGenderModel.from_pretrained(args.model_name).to(device)
    model.eval()

    if args.load_from_disk:
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)

    predict_fn = make_predict_gender_fn(processor, model, device, args.audio_path_column_name)
    dataset = dataset.map(
        predict_fn,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.cpu_num_workers,
    )

    print(f"Saving to {args.output_dir}...")
    dataset.save_to_disk(args.output_dir)
