import gradio as gr
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, pipeline, WhisperForConditionalGeneration, WhisperTokenizer, WhisperTokenizerFast
import numpy as np
import evaluate

# Example prompts from the paper (only style and text)
EXAMPLES = [
    [
        "A man speaks with a booming, medium-pitched voice in a clear environment, delivering his words at a measured speed.",
        "That's my brother. I do agree, though, it wasn't very well-groomed."
    ],
    [
        "A male speaker's speech is distinguished by a slurred articulation, delivered at a measured pace in a clear environment.",
        "reveal my true intentions in different ways. That's why the Street King Project and SMS"
    ],
    [
        "In a clear environment, a male speaker delivers his words hesitantly with a measured pace.",
        "the Grand Slam tennis game has sort of taken over our set that's sort of all the way"
    ],
    [
        "A low-pitched, guttural male voice speaks slowly in a clear environment.",
        "you know you want to see how far you can push everything and as an artist"
    ],
    [
        "A man speaks with a measured pace in a clear environment, displaying a distinct British accent.",
        "most important but the reaction is very similar throughout the world it's really very very similar"
    ],
    [
        "A male speaker's voice is clear and delivered at a measured pace in a quiet environment. His speech carries a distinct Jamaican accent.",
        "about God and the people him come from is more Christian, you know. We always"
    ],
    [
        "In a clear environment, a male voice speaks with a sad tone.",
        "Was that your landlord?"
    ],
    [
        "A man speaks with a measured pace in a clear environment, his voice carrying a sleepy tone.",
        "I mean, to be fair, I did see a UFO, so, you know."
    ],
    [
        "A frightened woman speaks with a clear and distinct voice.",
        "Yes, that's what they said. I don't know what you're getting done. What are you getting done? Oh, okay. Yeah."
    ],
    [
        "A woman speaks slowly in a clear environment, her voice filled with awe.",
        "Oh wow, this music is fantastic. You play so well. I could just sit here."
    ],
    [
        "A woman speaks with a high-pitched voice in a clear environment, conveying a sense of anxiety.",
        "this is just way too overwhelming. I literally don't know how I'm going to get any of this done on time. I feel so overwhelmed right now. No one is helping me. Everyone's ignoring my calls and my emails. I don't know what I'm supposed to do right now."
    ],
    [
        "A female speaker's high-pitched voice is clear and carries over a laughing, unobstructed environment.",
        "What is wrong with him, Chad?"
    ],
    [
        "In a clear environment, a man speaks in a whispered tone.",
        "The fruit piece, the still lifes, you mean."
    ],
    [
        "A male speaker with a husky, low-pitched voice delivers clear speech in a quiet environment.",
        "Ari had to somehow be subservient to Lloyd that would be unbelievable like if Lloyd was the guy who was like running Time Warner you know what I mean like"
    ],
    [
        "A female speaker's voice is clear and expressed at a measured pace, but carries a high-pitched, nasal tone, recorded in a quiet environment.",
        "You know, Joe Bow, hockey mom from Wasilla, if I have an idea that would perhaps make"
    ]
]

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

class ParlerTTSInference:
    def __init__(self):
        self.model = None
        self.description_tokenizer = None
        self.transcription_tokenizer = None
        self.asr_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_models(self, model_name, asr_model):
        """Load TTS and ASR models"""
        try:
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device)
            self.description_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transcription_tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            self.asr_pipeline = pipeline(model=asr_model, device=self.device, chunk_length_s=25.0)
            return gr.Button(value="🎵 Generate", variant="primary", interactive=True), "Models loaded successfully! You can now generate audio."
        except Exception as e:
            return gr.Button(value="🎵 Generate", variant="primary", interactive=False), f"Error loading models: {str(e)}"
    
    def generate_audio(self, description, text, guidance_scale, num_retries, wer_threshold):
        """Generate audio from text with style description"""
        if not all([self.model, self.description_tokenizer, self.transcription_tokenizer, self.asr_pipeline]):
            return None, "Please load the models first!"
        
        try:
            # Prepare inputs
            input_description = description.replace('\n', ' ').rstrip()
            input_transcription = text.replace('\n', ' ').rstrip()

            input_description_tokenized = self.description_tokenizer(input_description, return_tensors="pt").to(self.device)
            input_transcription_tokenized = self.transcription_tokenizer(input_transcription, return_tensors="pt").to(self.device)

            # Generate with ASR-based resampling
            generated_audios = []
            word_errors = []
            for i in range(num_retries):
                generation = self.model.generate(
                    input_ids=input_description_tokenized.input_ids,
                    prompt_input_ids=input_transcription_tokenized.input_ids,
                    guidance_scale=guidance_scale
                )
                audio_arr = generation.cpu().numpy().squeeze()

                word_error = wer(self.asr_pipeline, input_transcription, audio_arr, self.model.config.sampling_rate)

                if word_error < wer_threshold:
                    break
                generated_audios.append(audio_arr)
                word_errors.append(word_error)
            else:
                # Pick the audio with the lowest WER
                audio_arr = generated_audios[word_errors.index(min(word_errors))]
            
            return (self.model.config.sampling_rate, audio_arr), "Audio generated successfully!"
        except Exception as e:
            return None, f"Error generating audio: {str(e)}"

def create_demo():
    # Initialize the inference class
    inference = ParlerTTSInference()
    
    # Create the interface with a simple theme
    theme = gr.themes.Default()
    
    with gr.Blocks(title="ParaSpeechCaps Demo", theme=theme) as demo:
        gr.Markdown(
            """
            # 🎙️ Parler-TTS Mini with ParaSpeechCaps
            
            Generate expressive speech with rich style control using our Parler-TTS model finetuned on ParaSpeechCaps. Control various aspects of speech including:
            - Speaker characteristics (pitch, clarity, etc.)
            - Emotional qualities
            - Speaking style and rhythm
            
            Choose between two models:
            - **Full Model**: Trained on complete ParaSpeechCaps dataset
            - **Base Model**: Trained only on human-annotated ParaSpeechCaps-Base
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main settings
                model_name = gr.Dropdown(
                    choices=[
                        "ajd12342/parler-tts-mini-v1-paraspeechcaps",
                        "ajd12342/parler-tts-mini-v1-paraspeechcaps-only-base"
                    ],
                    value="ajd12342/parler-tts-mini-v1-paraspeechcaps",
                    label="Model",
                    info="Choose between the full model or base-only model"
                )
                
                description = gr.Textbox(
                    label="Style Description",
                    placeholder="Example: In a clear environment, a male voice speaks with a sad tone.",
                    lines=3
                )
                
                text = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=3
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    guidance_scale = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=1.5,
                        step=0.1,
                        label="Guidance Scale",
                        info="Controls the influence of the style description"
                    )
                    
                    num_retries = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Number of Retries",
                        info="Maximum number of generation attempts (for ASR-based resampling)"
                    )
                    
                    wer_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=50.0,
                        value=20.0,
                        step=1.0,
                        label="WER Threshold",
                        info="Word Error Rate threshold for accepting generated audio"
                    )
                    
                    asr_model = gr.Dropdown(
                        choices=["distil-whisper/distil-large-v2"],
                        value="distil-whisper/distil-large-v2",
                        label="ASR Model",
                        info="ASR model used for quality assessment"
                    )
                
                with gr.Row():
                    load_button = gr.Button("📥 Load Models", variant="primary")
                    generate_button = gr.Button("🎵 Generate", variant="primary", interactive=False)
                
            with gr.Column(scale=1):
                output_audio = gr.Audio(label="Generated Speech", type="numpy")
                status_text = gr.Textbox(label="Status", interactive=False)
        
        # Set up event handlers
        load_button.click(
            fn=inference.load_models,
            inputs=[model_name, asr_model],
            outputs=[generate_button, status_text]
        )
        
        def generate_with_default_params(description, text):
            return inference.generate_audio(
                description, text,
                guidance_scale=1.5,
                num_retries=3,
                wer_threshold=20.0
            )
        
        generate_button.click(
            fn=inference.generate_audio,
            inputs=[
                description,
                text,
                guidance_scale,
                num_retries,
                wer_threshold
            ],
            outputs=[output_audio, status_text]
        )
        
        # Add examples (only style and text)
        gr.Examples(
            examples=EXAMPLES,
            inputs=[
                description,
                text
            ],
            outputs=[output_audio, status_text],
            fn=generate_with_default_params,
            cache_examples=False
        )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)