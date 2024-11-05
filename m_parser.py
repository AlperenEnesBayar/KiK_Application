import os
from glob import glob
import textract

# Function to extract text from files using textract
def extract_text(file_path):
    try:
        text = textract.process(file_path).decode('utf-8')
        return text
    except Exception as e:
        print(f"Error extracting text from file: {file_path} -> {e}")
        return ""

def save_as_txt(file_path, text, base_input_dir):
    # Create the output folder structure by preserving the input folder hierarchy
    output_folder = "output_txt"
    relative_path = os.path.relpath(file_path, base_input_dir)  # Get relative path from base input dir
    txt_file_path = os.path.join(output_folder, relative_path + ".txt")  # Append .txt to the relative path

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(txt_file_path), exist_ok=True)

    # Save the extracted text
    try:
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text saved to: {txt_file_path}")
    except Exception as e:
        print(f"Error saving file: {txt_file_path} -> {e}")

# Base directory for input files
base_input_dir = "Akıllı Arama Mevzuat_v2"

# Search for all files in the given directory
files = glob(os.path.join(base_input_dir, "**/*"), recursive=True)
files = [file for file in files if os.path.isfile(file)]

# Process each file and extract text
for file in files:
    extension = file.split('.')[-1].lower()
    
    # Extract text using textract
    if extension in ['doc', 'docx', 'pdf']:
        text = extract_text(file)
        # Save the extracted text with preserved folder structure
        save_as_txt(file, text, base_input_dir)
    else:
        print(f"Unsupported file type: {file}")
