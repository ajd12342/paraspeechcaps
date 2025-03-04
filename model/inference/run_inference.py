import argparse
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, pipeline, WhisperForConditionalGeneration, WhisperTokenizer, WhisperTokenizerFast
import soundfile as sf
import evaluate

def wer(asr_pipeline, prompt, audio, sampling_rate):
    """
    Calculate Word Error Rate (WER) for a single audio sample against a reference text.
    Args:
        asr_pipeline: Huggingface ASR pipeline
        prompt: Reference text string
        audio: Audio array
        sampling_rate: Audio sampling rate
    
    Returns:
        float: Word Error Rate as a percentage
    """
    metric = evaluate.load("wer")

    # Handle Whisper's return_language parameter
    return_language = None
    if isinstance(asr_pipeline.model, WhisperForConditionalGeneration):
        return_language = True

    # Transcribe audio
    transcription = asr_pipeline(
        {"raw": audio, "sampling_rate": sampling_rate},
        return_language=return_language,
    )

    # Get appropriate normalizer
    if isinstance(asr_pipeline.tokenizer, (WhisperTokenizer, WhisperTokenizerFast)):
        tokenizer = asr_pipeline.tokenizer
    else:
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    english_normalizer = tokenizer.normalize
    basic_normalizer = tokenizer.basic_normalize

    # Choose normalizer based on detected language
    normalizer = (
        english_normalizer
        if isinstance(transcription.get("chunks", None), list) 
        and transcription["chunks"][0].get("language", None) == "english"
        else basic_normalizer
    )

    # Calculate WER
    norm_pred = normalizer(transcription["text"])
    norm_ref = normalizer(prompt)
    
    return 100 * metric.compute(predictions=[norm_pred], references=[norm_ref])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ParlerTTS inference with CFG and ASR-based resampling')
    parser.add_argument('--model_name', type=str, default="ajd12342/parler-tts-mini-v1-paraspeechcaps",
                        help='Name or path of the ParlerTTS model')
    parser.add_argument('--description', type=str, required=True,
                        help='Description of the speech style')
    parser.add_argument('--text', type=str, required=True,
                        help='Text to be synthesized')
    parser.add_argument('--guidance_scale', type=float, default=1.5,
                        help='Classifier-free guidance scale')
    parser.add_argument('--num_retries', type=int, default=3,
                        help='Number of retries for ASR-based resampling')
    parser.add_argument('--wer_threshold', type=float, default=20.0,
                        help='Word Error Rate threshold for ASR-based resampling')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the output audio file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run inference on (e.g., cuda, cpu)')
    parser.add_argument('--asr_model', type=str, default='distil-whisper/distil-large-v2',
                        help='Name or path of the ASR model for resampling')
    return parser.parse_args()

def main():
    args = parse_args()

    # Set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model and tokenizers
    model = ParlerTTSForConditionalGeneration.from_pretrained(args.model_name).to(device)
    description_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    transcription_tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")

    # Prepare inputs
    input_description = args.description.replace('\n', ' ').rstrip()
    input_transcription = args.text.replace('\n', ' ').rstrip()

    input_description_tokenized = description_tokenizer(input_description, return_tensors="pt").to(device)
    input_transcription_tokenized = transcription_tokenizer(input_transcription, return_tensors="pt").to(device)

    # Load ASR pipeline
    asr_pipeline = pipeline(model=args.asr_model, device=device, chunk_length_s=25.0)

    # Generate with ASR-based resampling
    generated_audios = []
    word_errors = []
    for i in range(args.num_retries):
        generation = model.generate(
            input_ids=input_description_tokenized.input_ids,
            prompt_input_ids=input_transcription_tokenized.input_ids,
            guidance_scale=args.guidance_scale
        )
        audio_arr = generation.cpu().numpy().squeeze()

        word_error = wer(asr_pipeline, input_transcription, audio_arr, model.config.sampling_rate)

        if word_error < args.wer_threshold:
            break
        generated_audios.append(audio_arr)
        word_errors.append(word_error)
    else:
        # Pick the audio with the lowest WER
        audio_arr = generated_audios[word_errors.index(min(word_errors))]

    # Save the output
    sf.write(args.output_file, audio_arr, model.config.sampling_rate)

if __name__ == "__main__":
    main() 