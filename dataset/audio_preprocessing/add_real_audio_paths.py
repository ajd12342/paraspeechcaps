import argparse
from pathlib import Path
from datasets import load_dataset

def add_audio_paths(example, source_to_root, validate_exists=False):
    """Adds real audio path to a single example."""
    source = example["source"]
    if source not in source_to_root:
        raise ValueError(f"Unknown source dataset: {source}")
    
    audio_path = str(Path(source_to_root[source]) / example["relative_audio_path"])
    
    if validate_exists and not Path(audio_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    return {
        "audio_path": audio_path
    }

def main():
    parser = argparse.ArgumentParser(description="Add real audio paths to dataset")
    parser.add_argument("--sources", nargs="+", required=True, 
                        help="List of source dataset names")
    parser.add_argument("--root_dirs", nargs="+", required=True,
                        help="List of root directories corresponding to sources")
    parser.add_argument("--dataset", required=True, 
                        help="HuggingFace dataset name")
    parser.add_argument("--save_mode", choices=["disk", "hub"], required=True,
                        help="Where to save the dataset")
    parser.add_argument("--output_path", required=True,
                        help="Output path (local directory or hub repo name)")
    parser.add_argument("--private", action="store_true",
                        help="If saving to hub, whether to make it private")
    parser.add_argument("--validate_exists", action="store_true",
                        help="Check if audio files exist during mapping")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for dataset mapping")
    
    args = parser.parse_args()
    
    if len(args.sources) != len(args.root_dirs):
        raise ValueError("Number of sources must match number of root directories")
    
    source_to_root = {
        source: root_dir
        for source, root_dir in zip(args.sources, args.root_dirs)
    }
    
    dataset = load_dataset(args.dataset)
    processed_dataset = dataset.map(
        lambda x: add_audio_paths(x, source_to_root, args.validate_exists),
        num_pro=args.num_workers,
        desc="Adding real audio paths"
    )
    
    if args.save_mode == "disk":
        processed_dataset.save_to_disk(args.output_path)
        print(f"Dataset saved to disk at {args.output_path}")
    elif args.save_mode == "hub":
        processed_dataset.push_to_hub(
            args.output_path,
            private=args.private
        )
        print(f"Dataset pushed to hub at {args.output_path}")
    else:
        raise ValueError(f"Invalid save mode: {args.save_mode}")

if __name__ == "__main__":
    main()
