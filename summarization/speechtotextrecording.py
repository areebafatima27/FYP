import whisper
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
from colorama import Fore, init

init(autoreset=True)

def split_audio_into_chunks(audio_path, output_folder, min_silence_len=700, silence_thresh=-40):
    """Split audio into smaller chunks based on silence."""
    print(Fore.CYAN + "Splitting audio into chunks...")
    try:
        sound = AudioSegment.from_wav(audio_path)
        chunks = split_on_silence(
            sound,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
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
    """Process each audio chunk with the Whisper model."""
    print(Fore.CYAN + "Processing audio chunks with Whisper model...")
    complete_transcription = ""

    for chunk_file in chunk_files:
        print(Fore.YELLOW + f"Processing: {chunk_file}")
        try:
            result = model.transcribe(chunk_file, task="translate")
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

# Paths
mp3_file_path = "D:\\Speech-to-Text-Urdu-English-Translator-main\\Speech-to-Text-Urdu-English-Translator-main\\audiomix.mp3"
wav_file_path = "D:\\Speech-to-Text-Urdu-English-Translator-main\\Speech-to-Text-Urdu-English-Translator-main\\audiomix.wav"
chunks_folder = "D:\\Speech-to-Text-Urdu-English-Translator-main\\chunks"
transcription_file_path = "D:\\Speech-to-Text-Urdu-English-Translator-main\\transcription.txt"

# Convert MP3 to WAV
if os.path.exists(mp3_file_path):
    print(Fore.CYAN + "Converting MP3 to WAV...")
    try:
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format="wav")
        print(Fore.GREEN + f"Converted {mp3_file_path} to {wav_file_path}")
    except Exception as e:
        print(Fore.RED + f"Error during MP3 to WAV conversion: {e}")
else:
    print(Fore.RED + "MP3 file not found!")
    exit()

# Initialize Whisper model
print(Fore.CYAN + "Loading Whisper model...")
try:
    model = whisper.load_model("medium")  # Choose 'tiny', 'base', 'small', 'medium', or 'large'
    print(Fore.GREEN + "Whisper model loaded successfully.")
except Exception as e:
    print(Fore.RED + f"Error loading Whisper model: {e}")
    exit()

# Split and Process Audio
if os.path.exists(wav_file_path):
    chunk_files = split_audio_into_chunks(wav_file_path, chunks_folder)
    if chunk_files:
        transcription = process_audio_chunks_with_whisper(chunk_files, model)
        if transcription.strip():
            # Save transcription to text file
            save_transcription_to_file(transcription, transcription_file_path)
        else:
            print(Fore.RED + "No transcription generated.")
    else:
        print(Fore.RED + "No chunks were created for processing.")
else:
    print(Fore.RED + "WAV file not found!")
