import os
import re
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import imutils
import tempfile
from PIL import Image
import io
import openai
import pandas as pd
import pygsheets
from dotenv import load_dotenv
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
import json
import sys
import requests

load_dotenv()
client = openai.OpenAI()
service_account_file=os.getenv('SECRET_KEY')
gc = pygsheets.authorize(service_file=service_account_file)
credentials = service_account.Credentials.from_service_account_file(service_account_file)
doc_ai_client = documentai.DocumentProcessorServiceClient(credentials=credentials)

def test_documentai(pdffile):
    try:
        # document OCR: 816d732010a931af
        # form: 8056980459f2eadc
        name = os.getenv('PROCESSOR_NAME')
        # Prepare request
        document = {"content": pdffile, "mime_type": "application/pdf"}
        request = {"name": name, "raw_document": document}

        # Make request
        result = doc_ai_client.process_document(request=request)
        print("Document processing complete.")

        # Return the full text of the document
        return result.document.text
    except Exception as e:
        print(f"Error with Document AI: {e}")
        return None

def upload_to_sheets(df, id_number):
    try:
        print("Attempting spreadsheet upload.")

        # Open spreadsheet
        sh = gc.open_by_url(os.getenv('SPREADSHEET'))
        
        # Name sheet with transcript id number
        new_sheet_name = f"{id_number}"

        # Check if a worksheet with this name already exists
        for wks in sh.worksheets():
            if wks.title == new_sheet_name:
                sh.del_worksheet(wks)
                print(f"Replacing old worksheet for {id_number}")
                break

        wks = sh.add_worksheet(new_sheet_name)

        # Update the worksheet with df, starting at cell A1 (1,1)
        wks.set_dataframe(df, (1,1))
        
        print("Spreadsheet uploaded successfully!")
    except Exception as e:
        print(f"Encountered error with spreadsheet upload: {e}")

def parse_table_to_df(api_output):
    try:
        print(api_output)
        print("Parsing table to df.")

        # Split the string into lines and remove empty lines
        lines = [line.strip() for line in api_output.strip().split('\n') if line.strip()]

        # Assuming the first line contains headers and the rest contain data
        data = [[cell.strip() for cell in line.split('|') if cell.strip()] for line in lines[2:]]

        # Create a DataFrame using the extracted headers and data
        df = pd.DataFrame(data, columns=['University', 'Year', 'Course Code', 'Title', 'Credits', 'Grade/Mark'])
        return df
    except Exception as e:
        print(f"Problem with dataframe: {e}")

# gpt-4o-2024-08-06
# gpt-4o-mini
def extrapolate_data(text):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """
                Task: Extract key course information from 2 inputs of text from a university transcript, denoted by 'INPUT 1:' and 'INPUT 2:'.
                Required Information:
                    University
                    Year
                    Course Code
                    Title
                    Credits
                    Grade/Mark
                Output Format: Present the extracted data in a tabular format with six column headers: ['University', 'Year', 'Course Code', 'Title', 'Credits', 'Grade/Mark']. Ensure that the output contains only these six columns and is devoid of any supplementary text or explanations, with no exception.
                Validation Step: After forming the table, review all entries to ensure that the values are contextually correct and logical. 
                """},
                {"role": "user", "content": f"{text}"}
            ]
        )
    except Exception as e:
        print(f"Error with GPT usage: {e}")

    return completion.choices[0].message.content

def perform_ocr_and_redact(doc, names_to_redact):
    """Perform OCR on each page of the PDF, extract text, apply redactions directly, and correct orientation."""
    new_doc = fitz.open()  # Create a new document for output

    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]

        zoom = 2  # Increase the resolution of the image for better OCR accuracy
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        # Extract the bytes of the image from the pixmap
        img_bytes = pix.tobytes("png")  # Save it in PNG format in memory

        # Convert bytes to a numpy array that cv2 can decode
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR) # Initially decode as color
        
        # Convert to grayscale for better orientation detection:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Use Tesseract to detect orientation and rotate if necessary
        osd = pytesseract.image_to_osd(gray)
        try:
          rotate_angle = float(re.search('(?<=Rotate: )\d+', osd).group(0))
          if rotate_angle != 0:
              # rotate the image to deskew it
              print(f"\tRotating page {page_num} by {rotate_angle} degrees")
              img = imutils.rotate_bound(img, rotate_angle)
        except:
          print("imutils problem")

        # now, actually run the OCR on np array after converting to gray
        img_np = np.array(img)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)  

        custom_config = r'--oem 3 --psm 1'
        d = pytesseract.image_to_data(img_gray, config=custom_config, output_type=pytesseract.Output.DICT)

        page_text = []
        # Loop over each word recognized by Tesseract
        for i in range(len(d['text'])):
            if int(d['conf'][i]) > 60:  # Confidence threshold
                text = d['text'][i].strip()
                if any(name.lower() == text.lower() for name in names_to_redact):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 0, 0), -1)  # Apply redaction on image
                    # rect = fitz.Rect(x, y, x + w, y + h)
                    # page.add_redact_annot(rect, text="REDACTED", fill=(0, 0, 0))
                else:
                    page_text.append(text) # adds the text to output if it doesn't contain the name

        # Append the page text to all_text
        all_text.append(" ".join(page_text))
        # Apply all redactions to the new page
        # page.apply_redactions()

        # Convert NumPy array back to PIL Image
        img = Image.fromarray(img_np)

        # Insert the corrected image back into the new PDF
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        new_page = new_doc.new_page(width=img.width, height=img.height)  # Create a new page with the same dimensions as the image
        new_page.insert_image(new_page.rect, stream=img_bytes)

    print('Succesfully completed OCR redaction')

    return new_doc, all_text  # Return the new document object with redactions and corrections

def save_txt_file(all_text, txt_output_path, filename):
    try:
        # Write collected text to a file
        output_txt_file = os.path.join(txt_output_path, f'{filename}_ocr_output.txt')
        with open(output_txt_file, 'w') as f:
            f.write("\n".join(all_text))
        print('Successfully created text file for debugging')
    except Exception as e:
        print(f'Error in text file: {e}')

def redact_names_in_pdf(pdf_data, filename):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(pdf_data)
            tmp_path = tmp.name

        # used for debugging:
        if os.path.getsize(tmp_path) == 0:
            print('Temporary file written, path:', tmp_path)
            print("Temporary file is empty after writing pdf_data")

        # Extract names from filename
        name_part, _ = filename.split('(')
        last_name, first_name = name_part.strip().split(', ')

        doc = fitz.open(tmp_path)

        # Pre-OCR Text Redaction
        for page in doc:
            # Accessing page dimensions
            page_rect = page.rect
            page_width = page_rect.width
            for name in [last_name, first_name]:
                text_instances = page.search_for(name)
                for inst in text_instances:
                    # Calculate full line bounding box
                    x0, y0, x1, y1 = inst.x0, inst.y0, inst.x1, inst.y1
                    full_line_bbox = fitz.Rect(0, y0, page_width, y1)
                    # Add redaction annotation covering the entire line
                    page.add_redact_annot(full_line_bbox, fill=(0, 0, 0))
            page.apply_redactions()
    except Exception as e:
        print(f"Failed with pre-OCR redaction due to: {e}")

    new_doc, all_text = perform_ocr_and_redact(doc, [last_name, first_name])  # Pass the names to be redacted
    my_string = " ".join(str(element) for element in all_text)

    return new_doc, my_string

def save_files(doc, output_path):
    try:
        # Use a temporary file that is automatically cleaned up
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf_path = temp_pdf.name
            doc.save(temp_pdf_path)
        
        # Save the redacted file
        
        print(f'Redacting file to: {output_path}')
        with open(temp_pdf_path, "rb") as file:
            with open(output_path, "wb") as output_file:
                output_file.write(file.read())
        print(f"File saved!")
        
    except Exception as e:
        print(f"Failed to save the file due to: {e}")
    finally:
        os.unlink(temp_pdf_path)

def process_files(input_folder, output_folder, txt_output_path, new_path):
    try:
        with open(new_path, 'w') as jsonl_file:
            for filename in os.listdir(input_folder):
                if filename.endswith(".pdf"):
                    match = re.search(r'\b\d{9}\b', filename)
                    if match:
                        id_number = match.group()
                        local_input_path = os.path.join(input_folder, filename)
                        local_output_path = os.path.join(output_folder, f"{id_number}.pdf")

                        # used for debugging:
                        print("-------------------------------------------")
                        print(f"Processing file: {filename}")
                        # print(f"Input: {local_input_path}")

                        with open(local_input_path, "rb") as file:
                            pdf_data = file.read()

                            try:
                                new_doc, text = redact_names_in_pdf(pdf_data, filename)
                                text = "INPUT 1: " + text
                                save_files(new_doc, local_output_path)
                                save_txt_file(text, txt_output_path, filename)
                            except Exception as e:
                                print(f"Error with manual OCR for {filename}: {e}")

                            try:
                                documentai_text = test_documentai(pdf_data)
                                if documentai_text:
                                    text = text + "\n\n INPUT 2: " + documentai_text
                            except Exception as e:
                                print("Error with DocumentAI usage. Using pytesseract text for extraction.")

                            if text:
                                request = {
                                    "custom_id": f"{filename}_trial",
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": "gpt-4o-2024-08-06",
                                        "messages": [
                                            {"role": "system", "content": "Extract the University, Year, Course Code, Title, Credits, Grade/Mark from university transcript data. Present the extracted data in a tabular format with six column headers: ['University', 'Year', 'Course Code', 'Title', 'Credits', 'Grade/Mark']. Ensure that the output contains only these six columns and is devoid of any supplementary text or explanations, with no exception."},
                                            {"role": "user", "content": f"{text}"}
                                        ]
                                    }
                                }
                                
                                # Write this request as a line in the .jsonl file
                                jsonl_file.write(json.dumps(request) + "\n")

                            else:
                                print(f"Error: No text for {filename}. Fatal error.")

                    else:
                        print(f"No valid ID found in filename: {filename}. Fatal error.")
    except Exception as e:
        print(f"Error in process_files: {e}")

def create_upload_batch(batch_file):
    batch_input_file = client.files.create(
    file=open(jsonl_input_path + f"/{batch_file}.jsonl", "rb"),
    purpose="batch"
    )

    batch_input_file_id = batch_input_file.id
    print(f"file_id: {batch_input_file_id}")

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": f"{batch_file}"
        }
    )

def download_openai_file(file_id, batch_name):
    api_key = os.getenv("OPENAI_API_KEY")
    url = f"https://api.openai.com/v1/files/{file_id}/content"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)
    batch_path = jsonl_results_path + f"/{batch_name}_output.jsonl"
    if response.status_code == 200:
        with open(batch_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully and saved as {batch_path}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        print(f"Response: {response.text}")

def process_batch_results(processed_name):
    # should already have _output
    new_path = jsonl_results_path + f"/{processed_name}.jsonl"
    with open(new_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            custom_id = data['custom_id']
            matched_id = re.search(r'\b\d{9}\b', custom_id).group()

            api_output = data['response']['body']['choices'][0]['message']['content']

            # Parse the API output into a DataFrame
            df = parse_table_to_df(api_output)  # Implement this function as you did before

            # Upload to Google Sheets
            upload_to_sheets(df, matched_id)

input_path = os.getenv('INPUT_PATH')
output_path = os.getenv('OUTPUT_PATH')
txt_output_path = os.getenv('TXT_OUTPUT_PATH')
jsonl_results_path = os.getenv('JSONL_RESULT_PATH')
jsonl_input_path = os.getenv('JSONL_INPUT_PATH')

def main():
    try:
        print("---------------")
        print("Select an option:")
        print("0 - Process files")
        print("1 - Upload the batch")
        print("2 - List the current batches")
        print("3 - Retrieve the batch results")
        print("4 - Process batch results")
        print("5 - Exit")

        print("\n")
        choice = input("Enter your choice: ")

        if choice == '0':
            try:
                batch_name = input("Enter the batch name: ")
                new_path = jsonl_input_path + f"/{batch_name}.jsonl"
                process_files(input_path, output_path, txt_output_path, new_path)
                print("---------------")
            except Exception as e:
                print(f"Error processing files: {e}")
        elif choice == '1':
            try: 
                batch_name = input("Enter the batch name to upload: ")
                create_upload_batch(batch_name)
                print("---------------")
            except Exception as e:
                print(f"Error uploading batch: {e}")
        elif choice == '2':
            try: 
                batches = client.batches.list(limit=10)
                for batch in batches.data:
                    print(f"ID: {batch.id}")
                    print(f"Status: {batch.status}")
                    print(f"Output File ID: {batch.output_file_id}")
                    print(f"Request Counts: {batch.request_counts}")
                    print("---------------")
            except Exception as e:
                print(f"Error connecting to OpenAI batch list: {e}")
        elif choice == '3':
            try:
                file_id = input("Paste file ID: ")
                batch_name = input("Enter the name of the batch: ")
                download_openai_file(file_id, batch_name)
                print("---------------")
            except Exception as e:
                print(f"Error downloading processed batch file: {e}")
        elif choice == '4':
            try:
                processed_name = input("Enter the name of the processed file. This should be a filename ending in _output. Type the NAME_output: ")
                process_batch_results(processed_name)
                print("---------------")
            except Exception as e:
                print(f"Error processing batch results: {e}")
        elif choice == '5':
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
