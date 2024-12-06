import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from transformers import pipeline
from colorama import Fore, init

init(autoreset=True)

def split_audio_into_chunks(audio_path, output_folder, min_silence_len=700, silence_thresh=-40):
    """Split audio into chunks based on silence."""
    print(Fore.CYAN + "Splitting audio into chunks...")
    try:
        sound = AudioSegment.from_wav(audio_path)
        chunks = split_on_silence(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(output_folder, f"chunk{i + 1}.wav")
            chunk.export(chunk_file, format="wav")
            chunk_files.append(chunk_file)
            print(Fore.GREEN + f"Exported: {chunk_file}")

        print(Fore.CYAN + f"Total {len(chunk_files)} chunks created.")
        return chunk_files
    except Exception as e:
        print(Fore.RED + f"Error during audio splitting: {e}")
        return []

def process_audio_chunks_with_whisper(chunk_files, model):
    """Transcribe audio chunks using the Whisper model."""
    print(Fore.CYAN + "Processing audio chunks with Whisper model...")
    complete_transcription = ""

    for chunk_file in chunk_files:
        print(Fore.YELLOW + f"Processing: {chunk_file}")
        try:
            result = model.transcribe(chunk_file, task="translate")  # task can be 'transcribe' or 'translate'
            recognized_text = result["text"]
            print(Fore.BLUE + f"Translated Text: {recognized_text}")
            complete_transcription += f"\n{recognized_text}"
        except Exception as e:
            print(Fore.RED + f"Error processing chunk {chunk_file}: {e}")

    return complete_transcription

def save_transcription_to_file(transcription, output_file_path):
    """Save the transcription to a text file."""
    print(Fore.CYAN + "Saving transcription to file...")
    try:
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(transcription)
        print(Fore.GREEN + f"Transcription saved to {output_file_path}")
    except Exception as e:
        print(Fore.RED + f"Error saving transcription: {e}")

def read_text_from_file(input_file):
    """Read text from a file."""
    with open(input_file, "r", encoding="utf-8") as file:
        return file.read()

def summarize_text(text, max_chunk_size=1024):
    """Summarize the provided text using Hugging Face's BART model."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    if len(text) <= max_chunk_size:
        print("Text is small enough to summarize directly...")
        chunks = [text]
    else:
        print("Text is too large, splitting into chunks for summarization...")
        chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    summary = []
    for chunk in chunks:
        input_length = len(chunk.split())
        max_length = min(300, max(50, int(input_length * 0.5)))  # Set summarization length dynamically
        summarized_chunk = summarizer(chunk, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
        summary.append(summarized_chunk)
    
    return ' '.join(summary)

def save_summary_to_file(summary_text, output_file):
    """Save the summary to a file."""
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(summary_text)
    print(Fore.GREEN + f"Summary saved to {output_file}")

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    """Convert an MP3 file to WAV format."""
    print(Fore.CYAN + "Converting MP3 to WAV...")
    try:
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format="wav")
        print(Fore.GREEN + f"Converted {mp3_file_path} to {wav_file_path}")
    except Exception as e:
        print(Fore.RED + f"Error during MP3 to WAV conversion: {e}")

def audio_to_summary(mp3_file_path, transcription_file_path, summary_file_path):
    """Complete workflow: convert audio to text and summarize it."""
    # Paths
    wav_file_path = mp3_file_path.replace(".mp3", ".wav")
    chunks_folder = "chunks"

    # Convert MP3 to WAV
    convert_mp3_to_wav(mp3_file_path, wav_file_path)

    # Initialize Whisper model
    print(Fore.CYAN + "Loading Whisper model...")
    try:
        model = whisper.load_model("medium")  # Choose 'tiny', 'base', 'small', 'medium', or 'large'
        print(Fore.GREEN + "Whisper model loaded successfully.")
    except Exception as e:
        print(Fore.RED + f"Error loading Whisper model: {e}")
        return

    # Split and Process Audio
    chunk_files = split_audio_into_chunks(wav_file_path, chunks_folder)
    if chunk_files:
        transcription = process_audio_chunks_with_whisper(chunk_files, model)
        if transcription.strip():
            # Save transcription to text file
            save_transcription_to_file(transcription, transcription_file_path)

            # Summarize the transcription
            summary_text = summarize_text(transcription)

            # Save the summary to a text file
            save_summary_to_file(summary_text, summary_file_path)
        else:
            print(Fore.RED + "No transcription generated.")
    else:
        print(Fore.RED + "No chunks were created for processing.")

if __name__ == "__main__":
    # Input paths (change these as needed)
    mp3_file_path = "ENGLISH.mp3"  # Audio file
    transcription_file_path = "transcription.txt"  # Transcription output
    summary_file_path = "summary.txt"  # Summary output

    # Run the complete audio to summary process
    audio_to_summary(mp3_file_path, transcription_file_path, summary_file_path)
