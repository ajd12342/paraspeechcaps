# Automatic Annotation — Intrinsic Tag Scaling

This directory contains the pipeline for scaling speaker-level intrinsic style tags (e.g., *shrill*, *guttural*, *husky*) from a small set of human-annotated speakers to a large target corpus using **perceptual speaker similarity** (VoxSim). This is described in Section 4.1 of the paper.

## Overview

The core idea: two speakers with high **perceptual** similarity (how similar humans *perceive* them to sound) tend to share most of their intrinsic style tags. We use VoxSim, a model trained specifically on human perceptual similarity ratings, to compute speaker embeddings and propagate intrinsic tags from annotated source speakers to similar target speakers.

```
Source speakers                Target corpus
(VoxCeleb, human-annotated     (e.g., Emilia, 45k hrs)
 with intrinsic tags)
        │                               │
        ▼                               ▼
  VoxSim embeddings              VoxSim embeddings
  (10 utterances/speaker)        (10 utterances/speaker)
        │                               │
        └──────── cosine similarity ────┘
                        │
              threshold ≥ 0.8
              (VoxSim score ≥ 5.0)
                        │
                        ▼
          Intrinsic tags propagated to
          all sufficiently similar
          target speakers
```

**18 intrinsic tags are propagated** (all except clarity-related tags):
Shrill, Nasal, Deep, Silky, Husky, Raspy, Guttural, Vocal-fry, Booming, Authoritative, Loud, Soft, Flowing, Monotonous, Punctuated, Hesitant, Singsong, Enunciated.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `create_manifest.py` | Sample N utterances per speaker → manifest TSV |
| 2 | `extract_embeddings.py` | Extract VoxSim embeddings from manifest |
| 3 | `create_intrinsic_tags_mapping.py` | Threshold by similarity → propagate tags |

Steps 1–2 must be run **twice**: once for the source (VoxCeleb) dataset and once for your target dataset (e.g., Emilia).

## Prerequisites

### 1. Source annotations
We provide `voxceleb_intrinsic_tags.json`, a JSON file containing human-annotated intrinsic tags for 511 VoxCeleb celebrities (paper Section 3). This is the source annotation file required by `create_intrinsic_tags_mapping.py`.

### 2. VoxSim pretrained model
Download the pretrained VoxSim model (`wavlm_ecapa.model`) from the [Google Drive link](https://drive.google.com/drive/folders/10YIeXdi1luhiwyUbKQFsm7nkH0h1lkkK?usp=drive_link) in the VoxSim trainer README. The relevant VoxSim files are vendored in the `voxsim/` subdirectory.

### 3. Source dataset (VoxCeleb)
You need audio files for the VoxCeleb celebrities that are annotated in `voxceleb_intrinsic_tags.json`. Follow the VoxCeleb setup instructions in [`../../README.md`](../../README.md) to download and preprocess the audio.

### 4. Installation
```bash
conda create -n paraspeechcaps-intrinsic python=3.11
conda activate paraspeechcaps-intrinsic
pip install torch torchaudio numpy scipy scikit-learn librosa soundfile s3prl tqdm datasets
```

> **Note:** `s3prl` is required by the VoxSim model to load `wavlm_large` features. On first run it will download WavLM Large weights from HuggingFace.

## Step-by-Step Usage

### Step 1: Create utterance manifests

Run `create_manifest.py` on **both** the source (VoxCeleb) and target datasets. The script samples `--num_utterances` (default: 10) utterances per speaker from all speakers with at least `--min_duration` seconds of audio (default: 300s = 5 min).

**Source (VoxCeleb) manifest:**

Your VoxCeleb dataset (after basic tags processing from Commit 1) has `audio_path`, `speakerid`, `name`, and `duration` columns.

```bash
python create_manifest.py \
    /path/to/voxceleb_hf_dataset \
    --load_from_disk \
    --output_path /path/to/manifests/voxceleb_manifest.tsv \
    --speaker_column speakerid \
    --name_column name \
    --min_duration 300 \
    --num_utterances 10
```

**Target dataset manifest** (e.g., Emilia after basic tags processing):

```bash
python create_manifest.py \
    /path/to/emilia_hf_dataset \
    --load_from_disk \
    --output_path /path/to/manifests/emilia_manifest.tsv \
    --speaker_column speakerid \
    --name_column name \
    --min_duration 300 \
    --num_utterances 10
```

**Output manifest TSV columns:** `audio_path`, `speaker`, `name`, `duration`

> Utterances are **sorted by speaker** in the output TSV. This ordering is required by `extract_embeddings.py` which groups embeddings by speaker based on row order.

### Step 2: Extract VoxSim embeddings

Run `extract_embeddings.py` on **both** manifests (source and target).

```bash
# Source (VoxCeleb) embeddings
python extract_embeddings.py \
    --initial_model /path/to/wavlm_ecapa.model \
    --audio_path_tsv /path/to/manifests/voxceleb_manifest.tsv \
    --output_dir /path/to/embeddings/voxceleb/

# Target (Emilia) embeddings
python extract_embeddings.py \
    --initial_model /path/to/wavlm_ecapa.model \
    --audio_path_tsv /path/to/manifests/emilia_manifest.tsv \
    --output_dir /path/to/embeddings/emilia/
```

**Output:**
- `embeddings_single.npy` — shape `(N_utterances, 256)`, one L2-normalized embedding per utterance in manifest order
- `audio_paths.txt` — list of audio paths matching the embedding rows
- Intermediate checkpoints `embeddings_single_<i>.npy` (saved every 1000 files by default)

> This step is GPU-intensive. On a single A100, expect ~0.5–1 second per utterance for `wavlm_large`.

### Step 3: Map intrinsic tags via perceptual similarity

```bash
python create_intrinsic_tags_mapping.py \
    --source_embeddings_npy /path/to/embeddings/voxceleb/embeddings_single.npy \
    --source_manifest_tsv   /path/to/manifests/voxceleb_manifest.tsv \
    --target_embeddings_npy /path/to/embeddings/emilia/embeddings_single.npy \
    --target_manifest_tsv   /path/to/manifests/emilia_manifest.tsv \
    --annotations_json      voxceleb_intrinsic_tags.json \
    --output_tags_json      /path/to/output/target_intrinsic_tags.json \
    --output_parent_speakers_json /path/to/output/target_parent_speakers.json
```

**Key parameters:**
- `--threshold 5.0` — VoxSim similarity threshold (1–6 scale). Default 5.0 corresponds to cosine similarity ≥ 0.8 (paper Section 4.1).
- `--num_utterances_per_speaker 10` — must match what was used in Step 1.
- `--min_annotations 2` — minimum annotator agreement for a tag to qualify.

**Output:**
- `target_intrinsic_tags.json` — `{"speaker_id": ["Husky", "Soft", ...], ...}`
- `target_parent_speakers.json` — `{"speaker_id": "Celebrity Name", ...}` (most similar source speaker)

## Output Format

The `target_intrinsic_tags.json` produced by this pipeline maps each target speaker ID to their propagated intrinsic tags:

```json
{
    "EMI_S12345": ["Husky", "Deep", "Authoritative"],
    "EMI_S67890": ["Shrill", "Nasal"],
    ...
}
```

This is passed to the style prompt generation step (see [`../../../style_prompts/`](../../../style_prompts/)) to generate natural language descriptions.

## Files

| File | Description |
|------|-------------|
| `create_manifest.py` | Sample utterances per speaker → manifest TSV |
| `extract_embeddings.py` | Extract VoxSim embeddings from manifest TSV |
| `create_intrinsic_tags_mapping.py` | Propagate intrinsic tags via similarity threshold |
| `voxceleb_intrinsic_tags.json` | Human-annotated intrinsic tags for 511 VoxCeleb celebrities |
| `voxsim/` | Vendored VoxSim model code (Ahn et al., Interspeech 2024) |

## Acknowledgements

The VoxSim embedding model and its code are from:

> Ahn et al., "[VoxSim: A Perceptual Voice Similarity Dataset](https://arxiv.org/abs/2406.06926)", Interspeech 2024.

The `voxsim/` directory contains files adapted from [kaistmm/voxsim_trainer](https://github.com/kaistmm/voxsim_trainer). Please cite VoxSim if you use this pipeline.
