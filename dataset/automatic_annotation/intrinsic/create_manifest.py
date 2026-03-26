"""
Create a per-speaker audio manifest for VoxSim embedding extraction.

For each speaker with sufficient total audio, samples up to --num_utterances
utterances and writes their audio paths to a TSV file. The output manifest is
passed to extract_embeddings.py.

IMPORTANT: Utterances for the same speaker must be contiguous in the manifest
because extract_embeddings.py uses the order of rows to group embeddings by
speaker when computing median speaker embeddings.

Input dataset format (HF dataset, from Commit 1 basic tags processing):
  - audio_path   (str)   absolute path to the .wav file
  - speakerid    (str)   unique speaker identifier
  - name         (str)   human-readable speaker name (e.g. celebrity name)
  - duration     (float) utterance duration in seconds

Output TSV columns:
  - audio_path   absolute path to the .wav file
  - speaker      speaker ID
  - name         speaker name
  - duration     utterance duration in seconds
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, load_from_disk


def main():
    parser = argparse.ArgumentParser(
        description="Create per-speaker audio manifest for VoxSim embedding extraction"
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="HF dataset name (Hub) or path to dataset saved with save_to_disk()",
    )
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        help="Load the dataset from disk using load_from_disk() instead of load_dataset()",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for the manifest TSV file",
    )
    parser.add_argument(
        "--audio_path_column",
        type=str,
        default="audio_path",
        help="Column name for audio file paths (default: audio_path)",
    )
    parser.add_argument(
        "--speaker_column",
        type=str,
        default="speakerid",
        help="Column name for speaker ID (default: speakerid)",
    )
    parser.add_argument(
        "--name_column",
        type=str,
        default="name",
        help="Column name for human-readable speaker name (default: name). "
             "Set to the same as --speaker_column if no separate name column exists.",
    )
    parser.add_argument(
        "--duration_column",
        type=str,
        default="duration",
        help="Column name for utterance duration in seconds (default: duration)",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=300.0,
        help="Minimum total speaker duration in seconds to include a speaker (default: 300)",
    )
    parser.add_argument(
        "--num_utterances",
        type=int,
        default=10,
        help="Number of utterances to sample per speaker (default: 10)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (e.g. 'train'). If not set, uses all data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for utterance sampling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load dataset
    if args.load_from_disk:
        dataset = load_from_disk(args.dataset_name)
    else:
        dataset = load_dataset(args.dataset_name)

    if args.split is not None:
        dataset = dataset[args.split]

    # Group examples by speaker
    speaker_to_examples = defaultdict(list)
    for example in dataset:
        speaker_id = example[args.speaker_column]
        speaker_to_examples[speaker_id].append(example)

    # Filter speakers by minimum total duration and sample utterances
    selected_rows = []
    for speaker_id, examples in speaker_to_examples.items():
        total_duration = sum(ex[args.duration_column] for ex in examples)
        if total_duration < args.min_duration:
            continue
        sampled = random.sample(examples, min(args.num_utterances, len(examples)))
        for ex in sampled:
            selected_rows.append(
                {
                    "audio_path": ex[args.audio_path_column],
                    "speaker": speaker_id,
                    "name": ex[args.name_column],
                    "duration": ex[args.duration_column],
                }
            )

    # Sort by speaker so all utterances for a speaker are contiguous — this is required
    # by extract_embeddings.py which groups embeddings by speaker based on row order.
    selected_rows.sort(key=lambda x: x["speaker"])

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "speaker", "name", "duration"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    num_speakers = len(set(row["speaker"] for row in selected_rows))
    print(
        f"Wrote {len(selected_rows)} utterances for {num_speakers} speakers to {args.output_path}"
    )


if __name__ == "__main__":
    main()
