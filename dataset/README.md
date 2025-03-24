# ParaSpeechCaps Dataset

This folder contains code related to the ParaSpeechCaps dataset, available on the HuggingFace Hub at [`ajd12342/paraspeechcaps`](https://huggingface.co/datasets/ajd12342/paraspeechcaps).

The dataset is licensed under the [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Table of Contents
1. [Installation](#1-installation)
   - [Setup Python environment](#11-setup-python-environment)
   - [Install dependencies](#12-install-dependencies)
2. [Usage](#2-usage)
   - [Load dataset annotations](#21-load-dataset-annotations)
   - [Processing dataset audio](#22-processing-dataset-audio)
3. [Dataset Structure](#3-dataset-structure)
4. [Dataset Statistics](#4-dataset-statistics)

## 1. Installation
### 1.1 Setup Python environment
This repository has been tested with Conda and Python 3.11. Other Python versions and package managers (`venv`, `uv`, etc.) should probably work.
```bash
conda create -n paraspeechcaps python=3.11
conda activate paraspeechcaps
```

### 1.2 Install dependencies
```bash
pip install datasets
```
To perform audio preprocessing to prepare dataset audio files, you will need to perform additional installation steps
```bash
# Install ffmpeg (https://ffmpeg.org/download.html) and sox (https://sourceforge.net/projects/sox/).
conda install conda-forge::ffmpeg conda-forge::sox
# Install voicefixer (https://github.com/haoheliu/voicefixer), pydub (https://github.com/jiaaro/pydub) and the HuggingFace CLI (https://huggingface.co/docs/huggingface_hub/en/guides/cli)
pip install git+https://github.com/haoheliu/voicefixer.git pydub huggingface_hub[cli]
```

## 2. Usage
The dataset provides style annotations and other metadata for each utterance in the dataset.

### 2.1 Load dataset annotations
You can inspect the dataset as follows:
```py
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
For ParaSpeechCaps-Base, we also provide full human intrinsic tag annotations for each example in the [`pscbase_name_to_intrinsictags.json`](./pscbase_name_to_intrinsictags.json) file, licensed under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), which is a JSON file mapping speaker names (in the `name` column of the main dataset) to a list of intrinsic tags for each example, with duplicates preserved (since multiple annotators can select the same tag). The only difference between the intrinsic tags in this file and the intrinsic tags in the `intrinsic_tags` column of the main dataset is that the dataset only contains tags that were selected by at least 2 annotators, while the full file contains all annotated tags.

### 2.2 Processing dataset audio
Our dataset provides style annotations and other metadata, but not the audio files themselves. Instead, it contains relative audio paths (in the `relative_audio_path` column). Here, we provide instructions on how to download the audio files from the respective source datasets (VoxCeleb, Expresso, EARS, Emilia) and process them.

You can decide where to download the audio files and place them for each source dataset; let's denote their root directories as `${voxceleb_root}`, `${expresso_root}`, `${ears_root}`, and `${emilia_root}` respectively.

#### 2.2.1 VoxCeleb
Request access to the [VoxCeleb dataset](https://mm.kaist.ac.kr/datasets/voxceleb/), download the audio files for both VoxCeleb1 and VoxCeleb2, and place them at `${voxceleb_root}` such that the directory structure is as follows:
```
${voxceleb_root}/
├── voxceleb1/
│   ├── dev/
│   │   └── wav/
│   └── test/
│       └── wav/
└── voxceleb2/
    ├── dev/
    │   └── aac/
    └── test/
        └── aac/
```
Convert the `.m4a` files in VoxCeleb2 to `.wav` files using the following script, which will create a copy of each audio file with a `.wav` extension in the same directory:
```bash
./audio_preprocessing/convert_m4a_to_wav.sh "${voxceleb_root}"/voxceleb2
```
Apply loudness normalization to all audio files using the following script, which will create a normalized copy of each `.wav` audio file overwriting the original file (the original file is saved with a `.backup` extension):
```bash
./audio_preprocessing/normalize_loudness.sh "${voxceleb_root}" # --show-total (optional, use to show total file count in progress bar, may be slower to start for large directories)
```
Apply Voicefixer noise removal to all audio files using the following script, which will create a `_voicefixer.wav` copy of each `.wav` audio file in the same directory:
```bash
python ./audio_preprocessing/apply_voicefixer.py "${voxceleb_root}"
```

#### 2.2.2 Expresso
Download the [Expresso dataset](https://github.com/facebookresearch/textlesslib/tree/main/examples/expresso/dataset) and place it at `${expresso_root}` such that the directory structure is as follows:
```
${expresso_root}/
├── README.txt
├── LICENSE.txt
├── read_transcriptions.txt
├── VAD_segments.txt
├── splits/
└── audio_48khz/
    ├── conversational/
    └── read/
```
Apply VAD segmentation to the Expresso conversational audio files, creating a `audio_48khz/conversational_vad_segmented` directory with the segmented audio files:
```bash
python ./audio_preprocessing/apply_expresso_vad.py "${expresso_root}"
```
Apply loudness normalization to all audio files using the following script, which will create a normalized copy of each `.wav` audio file overwriting the original file (the original file is saved with a `.backup` extension):
```bash
./audio_preprocessing/normalize_loudness.sh "${expresso_root}" # --show-total (optional, use to show total file count in progress bar, may be slower to start for large directories)
```

#### 2.2.3 EARS
Download the [EARS dataset](https://github.com/facebookresearch/ears_dataset) and place it at `${ears_root}` such that the directory structure is as follows:
```
${ears_root}/
├── p001/
├── p002/
├── ...
└── p107/
```
Apply loudness normalization to all audio files using the following script, which will create a normalized copy of each `.wav` audio file overwriting the original file (the original file is saved with a `.backup` extension):
```bash
./audio_preprocessing/normalize_loudness.sh "${ears_root}" # --show-total (optional, use to show total file count in progress bar, may be slower to start for large directories)
```

#### 2.2.4 Emilia
Download the [Emilia dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset) using the OpenDataLab format available at an older [revision](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07e8572f5f3be1dbd02ed3172a4d298f152) and place it at `${emilia_root}` such that the directory structure is as follows (exactly the same as the [original structure](https://huggingface.co/datasets/amphion/Emilia-Dataset?row=0#structure-on-opendatalab)):
```
${emilia_root}/
└── EN/
    ├── EN_B00000/
    ├── EN_B00001/
    ├── ...
    └── EN_B00113/
```
You can create this directory structure by untarring each `.tar.gz` file (e.g. `EN_B00000.tar.gz`) in the `EN` directory.

NOTE: If you use the HuggingFace `datasets` format to load the dataset instead of the OpenDataLab format, it will be a WebDataset with audios inside the dataset rather than a particular path, which makes it trickier to map the relative audio paths to real audio paths. In that case, you can use the fact that the filename (e.g. `EN_B00030_S01984_W000037.mp3`) in each `relative_audio_path` (e.g. `EN/EN_B00030/EN_B00030_S01984/mp3/EN_B00030_S01984_W000037.mp3`) is itself a unique identifier to perform the mapping; however, we have not tested this.

#### 2.2.5 Map relative audio paths to real audio paths
After downloading the audio files, placing them in the appropriate directories, and preprocessing them, mapping relative audio paths in the dataset to real audio paths is straightforward: simply append the relative audio path to the root directory for each source dataset (the dataset has a `source` column that specifies the source dataset for each row). Here is a helper script to do this that adds an `audio_path` column to the dataset:

```bash
# Save to disk
python ./audio_preprocessing/add_real_audio_paths.py \
    --sources voxceleb expresso ears emilia \
    --root_dirs "${voxceleb_root}" "${expresso_root}" "${ears_root}" "${emilia_root}" \
    --dataset ajd12342/paraspeechcaps \
    --save_mode disk \
    --output_path ./processed_dataset \
    --validate_exists # Optional: verify that all real audio paths exist

# Or save to the HuggingFace Hub 
python ./audio_preprocessing/add_real_audio_paths.py \
    --sources voxceleb expresso ears emilia \
    --root_dirs "${voxceleb_root}" "${expresso_root}" "${ears_root}" "${emilia_root}" \
    --dataset ajd12342/paraspeechcaps \
    --save_mode hub \
    --output_path your-username/paraspeechcaps-processed \
    --private \ # Optional: save as a private dataset, requires you to be logged in with `huggingface-cli login`
    --validate_exists # Optional: verify that all real audio paths exist
```

The `--validate_exists` flag, if provided, will check that each real audio path exists. This is useful to verify that all required audio files are present and properly organized after all preprocessing steps are complete.

After processing, you can load the dataset with the new `audio_path` column:
```python
# If saved to disk
dataset = load_dataset("./processed_dataset")

# If saved to the HuggingFace Hub as a private dataset
dataset = load_dataset("your-username/paraspeechcaps-processed")
```
NOTE: When loading a private dataset from the HuggingFace Hub, you'll need to be logged in with `huggingface-cli login`.

## 3. Dataset Structure

The dataset contains the following columns:

| Column | Type | Description |
|---------|------|-------------|
| source | string | Source dataset (e.g., Expresso, EARS, VoxCeleb, Emilia) |
| relative_audio_path | string | Relative path to identify the specific audio file being annotated |
| text_description | list of strings | 1-2 Style Descriptions for the utterance |
| transcription | string | Transcript of the speech |
| intrinsic_tags | list of strings | Tags tied to a speaker's identity (e.g., shrill, guttural) (null if non-existent) |
| situational_tags | list of strings | Tags that characterize individual utterances (e.g., happy, whispered) (null if non-existent) |
| basic_tags | list of strings | Basic tags (pitch, speed, gender, noise conditions) |
| all_tags | list of strings | Combination of all tag types |
| speakerid | string | Unique identifier for the speaker |
| name | string | Name of the speaker |
| duration | float | Duration of the audio in seconds |
| gender | string | Speaker's gender |
| accent | string | Speaker's accent (null if non-existent) |
| pitch | string | Description of the pitch level |
| speaking_rate | string | Description of the speaking rate |
| noise | string | Description of background noise |
| utterance_pitch_mean | float | Mean pitch value of the utterance |
| snr | float | Signal-to-noise ratio |
| phonemes | string | Phonetic transcription |

The `text_description` field is a list because each example may have 1 or 2 text descriptions:
- For Expresso and Emilia examples, all have 2 descriptions:
  - One with just situational tags
  - One with both intrinsic and situational tags
- For Emilia examples that were found by both our intrinsic and situational automatic annotation pipelines, there are 2 descriptions:
  - One with just intrinsic tags
  - One with both intrinsic and situational tags

The `relative_audio_path` field contains relative paths, functioning as a unique identifier for the specific audio file being annotated. The repository contains setup instructions that can properly link the annotations to the source audio files.

## 4. Dataset Statistics
The dataset covers a total of 59 style tags, including both speaker-level intrinsic tags (33) and utterance-level situational tags (26).
It consists of 282 train hours of human-labeled data and 2427 train hours of automatically annotated data (PSC-Scaled).
It contains 2518 train hours with intrinsic tag annotations and 298 train hours with situational tag annotations, with 106 hours of overlap.

| Split | Number of Examples | Number of Unique Speakers | Duration (hours) |
|-------|-------------------|-------------------------|------------------|
| train_scaled | 924,651 | 39,002 | 2,427.16 |
| train_base | 116,516 | 641 | 282.54 |
| dev | 11,967 | 624 | 26.29 |
| holdout | 14,756 | 167 | 33.04 |