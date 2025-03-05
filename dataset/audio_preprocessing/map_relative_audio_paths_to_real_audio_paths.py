import argparse
import csv
from pathlib import Path
from datasets import load_dataset

def get_real_audio_path(relative_path, source, root_dirs):
    """Maps a relative audio path to its real path based on the source dataset."""
    if source == "voxceleb":
        return str(Path(root_dirs["voxceleb"]) / relative_path)
    elif source == "expresso":
        return str(Path(root_dirs["expresso"]) / relative_path)
    elif source == "ears":
        return str(Path(root_dirs["ears"]) / relative_path)
    elif source == "emilia":
        return str(Path(root_dirs["emilia"]) / relative_path)
    else:
        raise ValueError(f"Unknown source dataset: {source}")

def main():
    parser = argparse.ArgumentParser(description="Map relative audio paths to real audio paths")
    parser.add_argument("--voxceleb_root", required=True, help="Root directory for VoxCeleb dataset")
    parser.add_argument("--expresso_root", required=True, help="Root directory for Expresso dataset")
    parser.add_argument("--ears_root", required=True, help="Root directory for EARS dataset")
    parser.add_argument("--emilia_root", required=True, help="Root directory for Emilia dataset")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--output_path", required=True, help="Output CSV file path")
    
    args = parser.parse_args()
    
    # Create dictionary of root directories
    root_dirs = {
        "voxceleb": args.voxceleb_root,
        "expresso": args.expresso_root,
        "ears": args.ears_root,
        "emilia": args.emilia_root
    }
    
    # Load all splits of the dataset
    dataset = load_dataset(args.dataset)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Field names for the CSV
    fieldnames = ['relative_audio_path', 'real_audio_path', 'source']
    
    # Write to CSV file
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each split
        for split in dataset.keys():
            split_data = dataset[split]
            
            # Process each example in the split
            for example in split_data:
                relative_path = example["relative_audio_path"]
                source = example["source"].lower()  # Convert to lowercase for consistency
                real_path = get_real_audio_path(relative_path, source, root_dirs)
                
                writer.writerow({
                    'relative_audio_path': relative_path,
                    'real_audio_path': real_path,
                    'source': source
                })
    
    print(f"Mapping saved to {output_path}")

if __name__ == "__main__":
    main()
