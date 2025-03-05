import voicefixer
from pathlib import Path
import sys
from pydub import AudioSegment, effects
import argparse

def apply_voicefixer(model, input_audio_path, output_audio_path):
    """Apply voicefixer to a single audio file and normalize the output."""
    if output_audio_path.exists():
        return
    
    model.restore(input=str(input_audio_path), output=str(output_audio_path), cuda=True, mode=0)
    # Normalize the audio using pydub
    output_audio = AudioSegment.from_file(output_audio_path, 'wav')
    output_audio = effects.normalize(output_audio, headroom=0.1)
    output_audio_path.unlink()  # Remove the file before writing normalized version
    output_audio.export(output_audio_path, format='wav')

def main():
    parser = argparse.ArgumentParser(description="Apply VoiceFixer to all WAV files in a directory")
    parser.add_argument('input_dir', type=Path, help='Directory containing WAV files to process')
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: {args.input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    model = voicefixer.VoiceFixer()
    print("Processing WAV files...")
    
    for input_path in args.input_dir.rglob("*.wav"):
        if input_path.stem.endswith('_voicefixer'):
            continue
            
        output_path = input_path.with_stem(f"{input_path.stem}_voicefixer")
        try:
            apply_voicefixer(model, input_path, output_path)
        except Exception as e:
            print(f"Failed to process {input_path}: {str(e)}", file=sys.stderr)

if __name__ == '__main__':
    main()
