import whisper

# Load the Whisper model
model = whisper.load_model("medium")  # You can also use 'base', 'small', 'large', etc.

# Transcribe the Urdu audio file
audio_path = "urduspeech.m4a"  # Replace with the path to your Urdu audio file
import whisper

# Load the Whisper model
model = whisper.load_model("medium")  # You can choose a different model size based on your needs

# Transcribe the Urdu audio file
audio_path = "urduspeech.m4a"  # Replace with your actual audio file path

# Perform transcription
result = model.transcribe(audio_path, language="ur")  # 'ur' for Urdu language

# Print the transcribed Urdu text
print("Transcribed Text in Urdu:")
print(result['text'])

# Perform transcription
result = model.transcribe(audio_path, language="ur")

# Print the transcribed Urdu text
print("Transcribed Text in Urdu:")
print(result['text'])
