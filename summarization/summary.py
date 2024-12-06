from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_chunk_size=1024):
    # Split text into chunks that fit within the model's max token length
    if len(text) <= max_chunk_size:
        print("Text is small enough to summarize directly...")
        chunks = [text]  # No need to split
    else:
        print("Text is too large, splitting into chunks for summarization...")
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    summary = []
    for chunk in chunks:
        # Dynamically set the max_length based on input length, ensuring it's less than the input
        input_length = len(chunk.split())
        max_length = min(300, max(50, int(input_length * 0.5)))  # 50% of input length or at least 50 tokens
        
        # Summarize each chunk
        summarized_chunk = summarizer(chunk, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']
        summary.append(summarized_chunk)
    
    # Combine all summarized chunks into one text
    summary_text = ' '.join(summary)
    return summary_text

def read_text_from_file(input_file):
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def save_summary_to_file(summary_text, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(summary_text)
    print(f"Summary saved to {output_file}")

if __name__ == "__main__":
    input_file = "smaller_input.txt"  # Replace with the path of your input text file
    output_file = "smaller_input_summary.txt"  # Replace with the desired output file name
    
    # Read text from the input file
    text = read_text_from_file(input_file)
    
    # Summarize the text
    summary_text = summarize_text(text)
    
    # Save the summarized text to the output file
    save_summary_to_file(summary_text, output_file)
    
    print("Summarization complete.")
