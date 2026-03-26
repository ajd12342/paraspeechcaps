# Basic Tag Extraction

This directory contains scripts for extracting basic acoustic and speaker tags: **gender**, **pitch**, **speaking rate**, and **noise level** (SNR). These correspond to the basic tags described in Section 3.2 of the paper.

The `dataspeech/` subdirectory contains the enrichment functions used by `extract_acoustic_values.py` and `metadata_to_text.py`. These are adapted from the [DataSpeech repository](https://github.com/huggingface/dataspeech) by HuggingFace (MIT License), with modifications to support `--load_from_disk` and our custom bin configuration.

## Directory Contents

```
basic_tags/
├── extract_acoustic_values.py  # Step 1: extract pitch, speaking rate, SNR from audio
├── extract_gender.py           # Step 2: extract gender predictions
├── metadata_to_text.py         # Step 3: bin continuous values to text labels
├── bin_edges.json              # ParaSpeechCaps bin thresholds (pitch/speed/noise)
├── text_bins.json              # Text labels for each bin
└── dataspeech/                 # Adapted from huggingface/dataspeech (MIT License)
    ├── cpu_enrichments/rate.py         # Speaking rate computation (g2p phonemes/sec)
    ├── gpu_enrichments/pitch.py        # Pitch estimation (PENN model)
    └── gpu_enrichments/snr_and_reverb.py  # SNR estimation (Brouhaha model)
```

## Input Format

The pipeline takes a HuggingFace dataset saved to disk. Each example must have at minimum:

| Column | Type | Description |
|--------|------|-------------|
| `audio_path` | string | Absolute path to the `.wav` audio file |
| `transcription` | string | Transcript of the speech |
| `speakerid` | string | Unique identifier for the speaker |
| `name` | string | Human-readable speaker name |

To create a HF dataset from a TSV metadata file:
```python
import datasets
ds = datasets.load_dataset(
    'csv',
    data_files={'train': 'metadata_train.tsv', 'dev': 'metadata_dev.tsv'},
    delimiter='\t'
)
ds.save_to_disk('./my-dataset')
```

## Pipeline Overview

```
HF Dataset → [Step 1: acoustic values] → [Step 2: gender] → [Step 3: text bins]
```

| Step | Script | Adds columns |
|------|--------|--------------|
| 1 | `extract_acoustic_values.py` | `utterance_pitch_mean`, `utterance_pitch_std`, `snr`, `c50`, `speaking_rate`, `phonemes` |
| 2 | `extract_gender.py` | `gender` ("male"/"female") |
| 3 | `metadata_to_text.py` | `pitch`, `noise`, `speaking_rate` (overwrites continuous value with text label), `reverberation`, `speech_monotony` |

## Prerequisites

```bash
pip install datasets[audio] transformers librosa torch
pip install penn                                          # pitch estimation
pip install https://github.com/marianne-m/brouhaha-vad/archive/main.zip  # SNR estimation
pip install g2p                                          # speaking rate (phoneme-based)
```

## Step 1: Extract Pitch, Speaking Rate, and SNR

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" python extract_acoustic_values.py \
    ./my-dataset \
    --load_from_disk \
    --output_dir ./my-dataset-step1 \
    --text_column_name "transcription" \
    --audio_path_column_name "audio_path" \
    --cpu_num_workers 16
```

## Step 2: Extract Gender

```bash
python extract_gender.py \
    --dataset_name ./my-dataset-step1 \
    --load_from_disk \
    --output_dir ./my-dataset-step2
```

This adds a `gender` column with values `"male"` or `"female"` (determined by comparing `male` vs. `female` class probabilities from the [audeering/wav2vec2-large-robust-24-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) model).

## Step 3: Bin Continuous Values to Text Labels

```bash
python metadata_to_text.py \
    ./my-dataset-step2 \
    --load_from_disk \
    --output_dir ./my-dataset-step3 \
    --cpu_num_workers 8 \
    --path_to_bin_edges ./bin_edges.json \
    --path_to_text_bins ./text_bins.json \
    --speaker_id_column_name "speakerid" \
    --gender_column_name "gender"
```

Pitch bins are computed **per gender** (using `bin_edges.json`'s `pitch_bins_male` and `pitch_bins_female`), which is why gender must be added before this step.

### Bin Configuration

The bin edges and text labels used in ParaSpeechCaps are in this directory:

- **`bin_edges.json`** — Thresholds for binning:
  - `speaking_rate`: 3 bins → slow / measured / fast (phonemes/sec)
  - `noise` (SNR in dB): 7 bins → very noisy → very clean
  - `pitch_bins_male` / `pitch_bins_female`: 3 bins each (Hz, gender-dependent)

- **`text_bins.json`** — Text labels for each bin:
  - `speaker_rate_bins`: `["slow speed", "measured speed", "fast speed"]`
  - `snr_bins`: `["very noisy environment", ..., "very clean environment"]`
  - `speaker_level_pitch_bins`: `["low-pitched", "medium-pitched", "high-pitched"]`

## Next Steps

After completing basic tag extraction, the dataset is ready for:
- [Intrinsic tag scaling](../intrinsic/) — scale speaker-level tags using VoxSim similarity
- [Situational tag scaling](../situational/) — scale utterance-level tags using DVA + semantic + Gemini
- [Style prompt generation](../../style_prompts/) — generate natural language descriptions with Mistral
