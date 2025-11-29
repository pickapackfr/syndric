from pypdf import PdfReader
import os
from pathlib import Path

# Directory containing PDF files
pdf_directory = Path("data/input")
output_directory = Path("data/output")

# Create output directory if it doesn't exist
output_directory.mkdir(parents=True, exist_ok=True)

# Get all PDF files in the directory
pdf_files = list(pdf_directory.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in data/input directory")
else:
    print(f"Found {len(pdf_files)} PDF file(s):")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")

    print("\n" + "=" * 50)

    # Process each PDF file
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        print("-" * 30)

        try:
            reader = PdfReader(pdf_file)

            # Print number of pages in this PDF
            print(f"Number of pages: {len(reader.pages)}")

            text = ""

            # Extract text from all pages
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += f"--- Page {page_num} ---\n"
                text += page_text
                text += "\n\n"

            # Create output filename (replace .pdf with .txt)
            output_filename = pdf_file.stem + ".txt"
            output_path = output_directory / output_filename

            # Save text to file
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(text)

            print(f"Extracted text saved to: {output_path}")
            print(f"Text length: {len(text)} characters")
            print("\n" + "=" * 50)

        except Exception as e:
            print(f"Error reading {pdf_file.name}: {e}")
            continue

    print(f"\nAll PDF files processed. Text files saved in: {output_directory}")
