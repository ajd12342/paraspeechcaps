"""
Map intrinsic style tags from annotated source speakers to target speakers via
perceptual speaker similarity (VoxSim).

For each annotated source speaker, we find target speakers whose median VoxSim
embedding has a cosine similarity >= --threshold with the source speaker's median
embedding, and copy the source speaker's intrinsic tags to those target speakers.
When a target speaker is similar to multiple source speakers, the most similar
source speaker's tags are used.

This implements the intrinsic tag scaling step from Section 4.1 of the paper:
  "For each VoxCeleb speaker, we find Emilia speakers that have a cosine similarity
  of at least 0.8 (corresponding to a similarity rating of 5 out of 6 in VoxSim)
  and copy all intrinsic tags (excluding clarity tags) from the VoxCeleb speaker
  to these Emilia speakers."

Inputs:
  --source_embeddings_npy : (N_src_utts, emb_dim) embeddings for source speakers
  --source_manifest_tsv   : TSV with 'name' column (one row per utterance, same
                            order as source embeddings; output of create_manifest.py
                            run on the source/VoxCeleb dataset)
  --target_embeddings_npy : (N_tgt_utts, emb_dim) embeddings for target speakers
  --target_manifest_tsv   : TSV with 'speaker' column (one row per utterance, same
                            order as target embeddings; output of create_manifest.py
                            run on the target dataset)
  --annotations_json      : JSON mapping source speaker name -> list of tags
                            (with duplicates preserved, i.e. a tag appearing twice
                            means two annotators agreed; see voxceleb_intrinsic_tags.json)
  --output_tags_json      : Output JSON mapping target speaker ID -> list of tags
  --output_parent_speakers_json : Output JSON mapping target speaker ID -> source
                            speaker name (the most similar source speaker)

Usage:
  python create_intrinsic_tags_mapping.py \\
      --source_embeddings_npy /path/to/voxceleb_embeddings/embeddings_single.npy \\
      --source_manifest_tsv   /path/to/voxceleb_manifest.tsv \\
      --target_embeddings_npy /path/to/emilia_embeddings/embeddings_single.npy \\
      --target_manifest_tsv   /path/to/emilia_manifest.tsv \\
      --annotations_json      voxceleb_intrinsic_tags.json \\
      --output_tags_json      /path/to/output/target_intrinsic_tags.json \\
      --output_parent_speakers_json /path/to/output/target_parent_speakers.json
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm


# Intrinsic tags that are propagated (excludes clarity-related tags, per paper Section 4.1)
INTRINSIC_TAGS_EXPANDABLE = [
    "Shrill", "Nasal", "Deep",
    "Silky", "Husky", "Raspy", "Guttural", "Vocal-fry",
    "Booming", "Authoritative", "Loud", "Soft",
    "Flowing", "Monotonous", "Punctuated", "Hesitant", "Singsong", "Enunciated",
]


def get_median_embeddings(single_embeddings, num_utterances_per_speaker, num_speakers):
    """Compute per-speaker median embedding, L2-normalised."""
    reshaped = single_embeddings.reshape((num_speakers, num_utterances_per_speaker, -1))
    median = np.median(reshaped, axis=1)
    median /= np.linalg.norm(median, axis=1, keepdims=True)
    return median


def main():
    parser = argparse.ArgumentParser(
        description="Map intrinsic tags from annotated source speakers to target speakers via VoxSim similarity"
    )
    parser.add_argument(
        "--source_embeddings_npy",
        type=str,
        required=True,
        help="Path to source (e.g. VoxCeleb) embeddings_single.npy "
             "(shape: num_source_utterances x emb_dim)",
    )
    parser.add_argument(
        "--source_manifest_tsv",
        type=str,
        required=True,
        help="Path to source manifest TSV with a 'name' column "
             "(output of create_manifest.py on the source dataset)",
    )
    parser.add_argument(
        "--target_embeddings_npy",
        type=str,
        required=True,
        help="Path to target (e.g. Emilia) embeddings_single.npy "
             "(shape: num_target_utterances x emb_dim)",
    )
    parser.add_argument(
        "--target_manifest_tsv",
        type=str,
        required=True,
        help="Path to target manifest TSV with a 'speaker' column "
             "(output of create_manifest.py on the target dataset)",
    )
    parser.add_argument(
        "--annotations_json",
        type=str,
        required=True,
        help="JSON file mapping source speaker name -> list of tags "
             "(with duplicates for multiple annotators; provided as voxceleb_intrinsic_tags.json)",
    )
    parser.add_argument(
        "--output_tags_json",
        type=str,
        required=True,
        help="Output JSON path for target speaker ID -> list of intrinsic tags",
    )
    parser.add_argument(
        "--output_parent_speakers_json",
        type=str,
        required=True,
        help="Output JSON path for target speaker ID -> most similar source speaker name",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="VoxSim similarity threshold (on the 1-6 scale: 5*(cos_sim)+1). "
             "Default: 5.0 corresponds to cosine similarity >= 0.8 (paper Section 4.1)",
    )
    parser.add_argument(
        "--max_similar_speakers",
        type=int,
        default=1000,
        help="Maximum number of similar target speakers per source speaker (default: 1000)",
    )
    parser.add_argument(
        "--num_utterances_per_speaker",
        type=int,
        default=10,
        help="Number of utterances per speaker in both manifests (default: 10)",
    )
    parser.add_argument(
        "--min_annotations",
        type=int,
        default=2,
        help="Minimum number of annotators that must agree on a tag to include it (default: 2)",
    )
    args = parser.parse_args()

    num_utt = args.num_utterances_per_speaker

    # ------------------------------------------------------------------
    # Load source embeddings and manifest
    # ------------------------------------------------------------------
    source_embeddings = np.load(args.source_embeddings_npy).squeeze()

    with open(args.source_manifest_tsv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        source_rows = list(reader)

    source_speakers = list(dict.fromkeys(row["name"] for row in source_rows))
    assert len(source_speakers) * num_utt == source_embeddings.shape[0], (
        f"Source: expected {len(source_speakers) * num_utt} rows in embeddings "
        f"(speakers × {num_utt}), got {source_embeddings.shape[0]}. "
        "Check --num_utterances_per_speaker and that the manifest matches the embeddings."
    )

    # ------------------------------------------------------------------
    # Load target embeddings and manifest; reorder so utterances per
    # speaker are contiguous (they should already be if create_manifest.py
    # was used, but we re-sort to be safe).
    # ------------------------------------------------------------------
    target_embeddings = np.load(args.target_embeddings_npy).squeeze()

    target_speaker_to_indices = defaultdict(list)
    with open(args.target_manifest_tsv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in tqdm.tqdm(enumerate(reader), desc="Loading target manifest"):
            target_speaker_to_indices[row["speaker"]].append(i)

    target_speakers = list(target_speaker_to_indices.keys())
    target_indices = []
    for spk in tqdm.tqdm(target_speakers, desc="Sorting target embeddings by speaker"):
        target_indices.extend(target_speaker_to_indices[spk])

    target_embeddings = target_embeddings[target_indices]
    assert len(target_speakers) * num_utt == target_embeddings.shape[0], (
        f"Target: expected {len(target_speakers) * num_utt} rows in embeddings "
        f"(speakers × {num_utt}), got {target_embeddings.shape[0]}. "
        "Check --num_utterances_per_speaker and that the manifest matches the embeddings."
    )

    # ------------------------------------------------------------------
    # Load annotations and filter to expandable tags with >= min_annotations
    # ------------------------------------------------------------------
    with open(args.annotations_json, "r") as f:
        celebrity2tags = json.load(f)

    # Only keep annotated speakers who appear in our source manifest
    source_speaker_set = set(source_speakers)
    celebrity2tags_filtered = {
        celeb: list(set(
            tag for tag in tags
            if tags.count(tag) >= args.min_annotations and tag in INTRINSIC_TAGS_EXPANDABLE
        ))
        for celeb, tags in celebrity2tags.items()
        if celeb in source_speaker_set
    }
    # Drop speakers with no qualifying tags
    celebrity2tags_filtered = {k: v for k, v in celebrity2tags_filtered.items() if v}
    annotated_speaker_set = set(celebrity2tags_filtered.keys())
    print(f"Annotated source speakers with qualifying tags: {len(annotated_speaker_set)}")

    # ------------------------------------------------------------------
    # Compute median embeddings and VoxSim similarity scores
    # ------------------------------------------------------------------
    print("Computing median source embeddings...")
    source_median = get_median_embeddings(source_embeddings, num_utt, len(source_speakers))

    print("Computing median target embeddings...")
    target_median = get_median_embeddings(target_embeddings, num_utt, len(target_speakers))

    print("Computing similarity scores...")
    # VoxSim similarity is reported on a 1-6 scale: score = 5*(cos_sim) + 1
    similarity_scores = 5.0 * (source_median @ target_median.T) + 1.0

    # ------------------------------------------------------------------
    # Find similar target speakers for each annotated source speaker
    # ------------------------------------------------------------------
    top_similar_speakers = []
    for i in tqdm.tqdm(range(len(source_speakers)), desc="Finding similar speakers"):
        if source_speakers[i] not in annotated_speaker_set:
            top_similar_speakers.append([])
            continue
        sorted_indices = np.argsort(similarity_scores[i])[::-1][: args.max_similar_speakers]
        similar = [
            (target_speakers[j], float(similarity_scores[i, j]))
            for j in sorted_indices
            if similarity_scores[i, j] > args.threshold
        ]
        top_similar_speakers.append(similar)

    # Filter to only annotated source speakers
    top_similar_filtered = [
        sim
        for i, sim in enumerate(top_similar_speakers)
        if source_speakers[i] in annotated_speaker_set
    ]
    names_filtered = [s for s in source_speakers if s in annotated_speaker_set]

    n_with_matches = sum(1 for sim in top_similar_filtered if len(sim) > 0)
    match_counts = [len(sim) for sim in top_similar_filtered if len(sim) > 0]
    print(f"Annotated speakers found in source embeddings:      {len(names_filtered)}")
    print(f"Annotated speakers with >= 1 similar target speaker: {n_with_matches}")
    if match_counts:
        print(f"Mean similar target speakers per annotated speaker:  {np.mean(match_counts):.1f}")
        print(f"Median similar target speakers per annotated speaker: {np.median(match_counts):.1f}")

    # ------------------------------------------------------------------
    # Assign each target speaker to the most similar source speaker
    # (resolve conflicts by keeping the highest similarity score)
    # ------------------------------------------------------------------
    target_to_best_source = {}
    for i, similar_speakers in enumerate(top_similar_filtered):
        source_name = names_filtered[i]
        for tgt_speaker, score in similar_speakers:
            if tgt_speaker not in target_to_best_source:
                target_to_best_source[tgt_speaker] = (source_name, score)
            else:
                _, current_score = target_to_best_source[tgt_speaker]
                if score > current_score:
                    target_to_best_source[tgt_speaker] = (source_name, score)

    print(f"Total new target speakers with intrinsic tags: {len(target_to_best_source)}")

    # Build output mappings
    target_to_tags = {
        spk: celebrity2tags_filtered[parent]
        for spk, (parent, _) in target_to_best_source.items()
    }
    target_to_parent = {
        spk: parent
        for spk, (parent, _) in target_to_best_source.items()
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    Path(args.output_tags_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tags_json, "w") as f:
        json.dump(target_to_tags, f, indent=4)
    print(f"Saved target speaker tags to {args.output_tags_json}")

    Path(args.output_parent_speakers_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_parent_speakers_json, "w") as f:
        json.dump(target_to_parent, f, indent=4)
    print(f"Saved target speaker parent mapping to {args.output_parent_speakers_json}")


if __name__ == "__main__":
    main()
