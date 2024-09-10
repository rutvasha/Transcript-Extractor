
# PDF Redaction and Data Extraction Pipeline

## Overview

This project processes PDF files to redact sensitive information, extract structured text data (e.g., from university transcripts), and upload the results to Google Sheets. It integrates multiple technologies, including OpenAI’s GPT models, Google Cloud services (Document AI and Google Sheets), and OCR tools like PyMuPDF and Tesseract. This pipeline is designed to handle sensitive data securely, process it efficiently, and generate structured outputs.

## Key Features

- **PDF Redaction**: Automatically redacts sensitive names and personal identifiers from PDF documents.
- **OCR (Optical Character Recognition)**: Uses Tesseract and Google Document AI to extract text from PDFs, ensuring accurate data extraction.
- **Structured Data Extraction**: Extracts specific data fields (e.g., University, Year, Course Code, Credits, Grades) from text using OpenAI GPT models.
- **Google Sheets Integration**: Uploads extracted and structured data to Google Sheets, creating or replacing worksheets based on transcript IDs.
- **Batch Processing**: Supports batch processing of multiple PDF files, submitting them to OpenAI for bulk extraction.

## Technologies Used

- **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/en/latest/)**: For reading, modifying, and redacting PDF files.
- **[OpenCV](https://opencv.org/)**: For image processing, including correcting page orientation before OCR.
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)**: For performing OCR and extracting text from PDF images.
- **[Google Cloud Document AI](https://cloud.google.com/document-ai)**: For advanced OCR, particularly useful for complex document layouts.
- **[OpenAI GPT](https://openai.com/)**: For extracting structured data from the text via natural language processing.
- **[Pygsheets](https://pygsheets.readthedocs.io/en/stable/)**: For interacting with Google Sheets API and uploading processed data.
- **[Pandas](https://pandas.pydata.org/)**: For structuring extracted data into tables.

## Pipeline Overview

1. **File Input**: 
   - PDF files are provided as input. Each file is expected to contain sensitive data (e.g., names) and structured data (e.g., university transcripts).
   - The file names are expected to contain the person's last and first name for redaction purposes.

2. **Redaction and OCR**:
   - **Redaction**: Names are extracted from the filename and redacted in the document using PyMuPDF.
   - **OCR**: After redaction, the document undergoes OCR using Tesseract and optionally Google Document AI for text extraction.

3. **Data Extraction**:
   - OpenAI GPT models are used to extract specific structured data (e.g., course details, grades) from the raw text.
   - The extracted data is presented in a tabular format.

4. **Google Sheets Upload**:
   - The extracted data is uploaded to a Google Sheet. Each transcript creates a new worksheet in the sheet, named according to its ID number.

5. **Batch Processing**:
   - The script can handle multiple PDFs in batch mode, submitting the files for processing and gathering the results in bulk.

## Setup

### Prerequisites

- Python 3.7 or higher
- Google Cloud credentials (for Document AI and Google Sheets API)
- OpenAI API key

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository-url.git
   cd your-repository-folder
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the project root with the following variables:
     ```bash
     SECRET_KEY=<your-service-account-key>
     SPREADSHEET=<your-google-sheets-url>
     INPUT_PATH=<path-to-pdf-input-files>
     OUTPUT_PATH=<path-to-save-redacted-files>
     TXT_OUTPUT_PATH=<path-to-save-extracted-text-files>
     JSONL_RESULT_PATH=<path-to-jsonl-output-files>
     JSONL_INPUT_PATH=<path-to-jsonl-input-files>
     OPENAI_API_KEY=<your-openai-api-key>
     ```

4. Set up Google Cloud Document AI and Google Sheets credentials:
   - Download the service account key file from Google Cloud and place it in your project directory.

### Running the Pipeline

1. To start processing files, run the script:
   ```bash
   python main.py
   ```

2. Follow the on-screen prompts to:
   - Process files
   - Upload batches to OpenAI
   - Retrieve and process results

### Example Usage

- Place your PDF files in the directory specified by `INPUT_PATH`.
- The script will automatically redact names, extract data, and upload the results to Google Sheets.

## Future Improvements

- Support for additional document types beyond university transcripts.
- Enhanced error handling for better fault tolerance.
- Further optimization for large-scale batch processing.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
