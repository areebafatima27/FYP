import whisper
import os
import sys

# Check if ffmpeg is installed
if not os.system("ffmpeg -version") == 0:
    print("Please install FFmpeg and add it to your PATH")
    sys.exit(1)

# Initialize the Whisper model
model = whisper.load_model("base")  # You can use 'small', 'medium', or 'large' depending on your needs

# Function to transcribe audio with automatic language detection
def transcribe_audio(audio_path):
    try:
        result = model.transcribe(audio_path)
        return result['text'], result['language']
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Main
if __name__ == "__main__":
    # Path to the audio file
    audio_file = r"C:\Users\Touch Smart\OneDrive\Desktop\test\audio\audio1.mp3"

    # Debugging: print the file path
    print(f"Looking for file: {audio_file}")
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"File not found: {audio_file}")
        sys.exit(1)

    # Transcribe the audio
    transcription, detected_language = transcribe_audio(audio_file)

    if transcription:
        print(f"\nDetected Language: {detected_language}")
        print("Transcription:")
        print(transcription)
    else:
        print("Transcription failed.")
