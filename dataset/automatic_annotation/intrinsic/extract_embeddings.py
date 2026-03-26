"""
Extract per-utterance VoxSim speaker embeddings from a manifest TSV.

Adapted from the VoxSim trainer repository by Ahn et al.
https://github.com/kaistmm/voxsim_trainer
Ahn et al., "VoxSim: A Perceptual Voice Similarity Dataset", Interspeech 2024.

The VoxSim model files are vendored in the voxsim/ subdirectory alongside this script.
Download the pretrained VoxSim model weights from the Google Drive link in the VoxSim
trainer README and pass the path via --initial_model.

Output:
  - embeddings_single.npy  : float32 array of shape (N, emb_dim), one embedding per
                             utterance, in the same order as the manifest TSV rows.
                             Intermediate checkpoints are saved as
                             embeddings_single_<i>.npy and then concatenated.
  - audio_paths.txt        : ordered list of audio paths matching embeddings_single.npy
  - args.txt               : the args used for this run

Usage:
  python extract_embeddings.py \\
      --initial_model /path/to/wavlm_ecapa.model \\
      --audio_path_tsv /path/to/manifest.tsv \\
      --output_dir /path/to/output/
"""

import csv
import os
import sys
from pathlib import Path

# Add the vendored voxsim/ directory to sys.path so SpeakerNet, DatasetLoader,
# models/, and loss/ are all importable without modification.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "voxsim"))

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from SpeakerNet import SpeakerNet, WrappedModel
from DatasetLoader import loadWAV


def load_parameters(speaker_model, gpu, path):
    self_state = speaker_model.module.state_dict()
    loaded_state = torch.load(path, map_location="cuda:%d" % gpu)
    if "model" in loaded_state:
        loaded_state = loaded_state["model"]
        newdict = {}
        delete_list = []
        for name, param in loaded_state.items():
            new_name = "__S__." + name
            newdict[new_name] = param
            delete_list.append(name)
        loaded_state.update(newdict)
        for name in delete_list:
            del loaded_state[name]

    copied_keys = set()
    for name, param in loaded_state.items():
        origname = name
        if name.startswith("speaker_encoder"):
            name = name.replace("speaker_encoder", "__S__")
        if name not in self_state:
            name = name.replace("module.", "")
            if name not in self_state:
                print("{} is not in the model.".format(origname))
                continue
        if self_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: {}, model: {}, loaded: {}".format(
                origname, self_state[name].size(), loaded_state[origname].size()))
            continue
        self_state[name].copy_(param)
        copied_keys.add(name)
    for name in self_state:
        if name not in copied_keys:
            print("Not copied: ", name)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, eval_frames, num_eval):
        self.audio_paths = audio_paths
        self.eval_frames = eval_frames
        self.num_eval = num_eval

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # loadWAV returns (feats, audio_org); we use audio_org (the full-file embedding)
        _, audio = loadWAV(self.audio_paths[idx], self.eval_frames,
                           evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract per-utterance VoxSim speaker embeddings"
    )
    parser.add_argument(
        "--initial_model",
        type=str,
        required=True,
        help="Path to pretrained VoxSim model weights (e.g. wavlm_ecapa.model). "
             "Download from the Google Drive link in the VoxSim trainer README.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wavlm_large",
        help="Model backbone (default: wavlm_large)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=256,
        help="Output embedding dimension (default: 256)",
    )
    parser.add_argument(
        "--eval_frames",
        type=int,
        default=0,
        help="Input length in frames; 0 uses the whole file (default: 0)",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        default=80,
        help="Number of mel filterbanks (default: 80)",
    )
    parser.add_argument(
        "--mlp",
        type=int,
        default=None,
        nargs="*",
        help="MLP layer sizes (optional)",
    )
    parser.add_argument(
        "--trainfunc",
        type=str,
        default="mse",
        help="Loss function used during training (default: mse)",
    )
    parser.add_argument(
        "--update_extract",
        type=bool,
        default=False,
        help="Whether to update the feature extractor (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed (default: 10)",
    )
    parser.add_argument(
        "--audio_path_tsv",
        type=str,
        required=True,
        help="Path to manifest TSV with an 'audio_path' column (output of create_manifest.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save embeddings_single.npy and metadata",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=1000,
        help="Save intermediate checkpoint every N files (default: 1000)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes (default: 4)",
    )

    args = parser.parse_args()
    args.eval = True
    args.gpu = 0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    s = SpeakerNet(**vars(args))
    s = WrappedModel(s).cuda(args.gpu)
    load_parameters(s, args.gpu, args.initial_model)
    s.eval()

    # Read audio paths from manifest
    wav_files = []
    with open(args.audio_path_tsv, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            wav_files.append(row["audio_path"])

    # Save audio path list for alignment with embeddings
    with open(Path(args.output_dir) / "audio_paths.txt", "w") as f:
        for wav_file in wav_files:
            f.write(wav_file + "\n")
    with open(Path(args.output_dir) / "args.txt", "w") as f:
        f.write(str(args))

    # Extract embeddings, saving intermediate checkpoints
    dataset = AudioDataset(wav_files, args.eval_frames, num_eval=10)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    embeddings_batch = []
    last_i = 0
    for i, audio in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        audio = audio.cuda()
        with torch.no_grad():
            emb = s.module.__S__.forward(audio)
            emb = F.normalize(emb, p=2, dim=1).detach().cpu().numpy()
        embeddings_batch.append(emb)
        last_i = i
        if i % args.checkpoint_every == args.checkpoint_every - 1:
            chunk = np.concatenate(embeddings_batch, axis=0)
            np.save(Path(args.output_dir) / f"embeddings_single_{i}.npy", chunk)
            embeddings_batch = []

    # Save final chunk
    if embeddings_batch:
        chunk = np.concatenate(embeddings_batch, axis=0)
        np.save(Path(args.output_dir) / f"embeddings_single_{last_i}.npy", chunk)

    # Concatenate all chunks into a single file
    chunk_files = sorted(
        Path(args.output_dir).glob("embeddings_single_*.npy"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    all_embeddings = np.concatenate([np.load(p) for p in chunk_files], axis=0)
    np.save(Path(args.output_dir) / "embeddings_single.npy", all_embeddings)
    print(
        f"Saved {all_embeddings.shape[0]} embeddings "
        f"(dim={all_embeddings.shape[1]}) to "
        f"{args.output_dir}/embeddings_single.npy"
    )
