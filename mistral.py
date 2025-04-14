import os
import json
import requests
from dotenv import load_dotenv
from mistralai import Mistral
import csv

# Load environment variables from .env file
load_dotenv()
api_key = os.environ["MISTRAL_API_KEY"]

# Initialize the Mistral client with your API key
client = Mistral(api_key=api_key)

# Google Drive sharing link
google_drive_link = "https://drive.google.com/file/d/1tmK0-vKl2exISii6u1WDUoa3T1sd0vBZ/view?usp=sharing"

# Function to convert Google Drive sharing link to direct download link
def get_direct_download_link(google_drive_link):
    file_id = google_drive_link.split('/d/')[1].split('/view')[0]
    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    return direct_link

# Get the direct download link
pdf_url = get_direct_download_link(google_drive_link)

# Ensure the PDF URL is publicly accessible
pdf_url = pdf_url.replace('uc?export=download', 'uc?export=view')

# Process the document with OCR using the PDF URL
try:
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": pdf_url
        },
        include_image_base64=True
    )
except Exception as e:
    print("Error processing OCR:", e)
    exit(1)

# Extract the text from the OCR response
ocr_text = ocr_response.pages[0].markdown

# Function to extract transactions from the OCR text
def extract_transactions(ocr_text):
    transactions = []
    lines = ocr_text.split('\n')
    table_detected = False

    for line in lines:
        parts = line.split('|')
        if len(parts) >= 8:
            date = parts[1].strip()
            description = parts[3].strip()
            amount = parts[5].strip()
            balance = parts[7].strip()
            transactions.append({
                "date": date,
                "description": description,
                "amount": amount,
                "balance": balance
            })
        elif "Tran Date" in parts and "Description" in parts and "Amount" in parts and "Balance" in parts:
            table_detected = True
            continue
        elif table_detected:
            # If a table is detected, convert it to structured JSON
            table_data = []
            for row in lines:
                row_parts = row.split('|')
                if len(row_parts) >= 8:
                    table_data.append({
                        "date": row_parts[1].strip(),
                        "description": row_parts[3].strip(),
                        "amount": row_parts[5].strip(),
                        "balance": row_parts[7].strip()
                    })
            return json.dumps(table_data, indent=2)

    return transactions

# Extract transactions from the OCR text
transactions = extract_transactions(ocr_text)

# Convert the transactions to JSON
transactions_json = json.dumps(transactions, indent=2)

# Print the JSON output
print(transactions_json)

# Convert the transactions to CSV
csv_filename = 'transactions.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["date", "description", "amount", "balance"])
    writer.writeheader()
    for transaction in transactions:
        writer.writerow(transaction)

print(f"CSV file '{csv_filename}' has been created.")