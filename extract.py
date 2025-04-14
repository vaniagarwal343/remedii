import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
import time

from dotenv import load_dotenv
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from unstract.llmwhisperer import LLMWhispererClientV2, LLMWhispererClientException


class CustomerAddress(BaseModel):
    zip_code: str = Field(description="The zip code alone")
    city: str = Field(description="The city name from the address")
    full_address: str = Field(description="The full address of the customer")


class PaymentInfo(BaseModel):
    due_date: datetime = Field(description="The due date of the credit card statement")
    minimum_payment: float = Field(description="The minimum amount due")
    new_balance: float = Field(description="The total new balance amount")


class SpendLineItem(BaseModel):
    spend_date: datetime = Field(description="The date of the transaction")
    spend_description: str = Field(description="The description of the spend")
    amount: float = Field(description="The amount of the transaction")


class ParsedCreditCardStatement(BaseModel):
    issuer_name: str = Field(description="Issuer or bank name")
    customer_name: str = Field(description="Name of the customer")
    customer_address: CustomerAddress = Field(description="Customer's address")
    payment_info: PaymentInfo = Field(description="Payment details")
    spend_line_items: list[SpendLineItem] = Field(description="List of spend items")


def make_llm_whisperer_call(file_path):
    print(f"Processing file: {os.path.abspath(file_path)}")

    api_key = os.getenv("LLM_WHISPERER_API_KEY")
    base_url = os.getenv("LLMWHISPERER_BASE_URL_V2", "https://llmwhisperer-api.us-central.unstract.com/api/v2")

    if not api_key:
        raise ValueError("Missing LLM_WHISPERER_API_KEY in environment variables.")
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    print(f"Using API key (redacted): {api_key[:4]}...{api_key[-4:]}")
    print(f"Base URL: {base_url}")
    print(f"Uploading file: {repr(file_path)}")

    client = LLMWhispererClientV2(api_key=api_key, base_url=base_url)

    try:
        result = client.whisper(file_path=file_path)
        if result.get("status_code") == 202:
            print(f"Whisper job accepted, hash: {result['whisper_hash']}")

            # Polling for status (async mode)
            while True:
                status = client.whisper_status(result["whisper_hash"])
                print(f"STATUS: {status['status']}")

                if status["status"] in ["delivered", "processed"]:
                    extraction = client.whisper_retrieve(result["whisper_hash"])
                    return extraction["extraction"]["result_text"]

                if status["status"] in ["unknown", "failed"]:
                    raise ValueError(f"Whisper processing failed with status: {status['status']}")

                time.sleep(5)  # Poll every 5 seconds
        elif result.get("status_code") == 200:
            return result["extraction"]["result_text"]

    except LLMWhispererClientException as e:
        raise ValueError(f"LLMWhisperer error: {e.message}, Status Code: {e.status_code}")


def generate_cache_file_name(file_path):
    if os.path.getsize(file_path) < 4096:
        error_exit("File too small to process.")
    with open(file_path, "rb") as f:
        first_block = f.read(4096)
        f.seek(-4096, os.SEEK_END)
        last_block = f.read(4096)
    return f"/tmp/{hashlib.md5(first_block).hexdigest()}_{hashlib.md5(last_block).hexdigest()}.txt"


def is_file_cached(file_path):
    return Path(generate_cache_file_name(file_path)).is_file()


def extract_text(file_path):
    if is_file_cached(file_path):
        print(f"Info: File {file_path} is already cached.")
        with open(generate_cache_file_name(file_path), "r") as f:
            return f.read()
    else:
        data = make_llm_whisperer_call(file_path)
        with open(generate_cache_file_name(file_path), "w") as f:
            f.write(data)
        return data


def error_exit(message):
    print(f"ERROR: {message}")
    sys.exit(1)


def show_usage_and_exit():
    error_exit("Please provide a directory or file to process.")


def enumerate_pdf_files(input_path):
    files_to_process = []
    if os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        files_to_process.append(input_path)
    elif os.path.isdir(input_path):
        for file_name in os.listdir(input_path):
            full_path = os.path.join(input_path, file_name)
            if os.path.isfile(full_path) and file_name.lower().endswith('.pdf'):
                files_to_process.append(full_path)
    else:
        error_exit(f"{input_path} is not a valid file or directory.")
    return files_to_process


def extract_values_from_file(raw_text):
    preamble = ("Extract and summarize information accurately from the credit card statement provided. "
                "Only use the information provided — do not use external knowledge.")
    postamble = "Do not include any explanation in the reply. Only return the extracted JSON."

    system_message = SystemMessagePromptTemplate.from_template("{preamble}")
    human_message = HumanMessagePromptTemplate.from_template("{format_instructions}\n{raw_text}\n{postamble}")

    parser = PydanticOutputParser(pydantic_object=ParsedCreditCardStatement)

    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    request = chat_prompt.format_prompt(
        preamble=preamble,
        format_instructions=parser.get_format_instructions(),
        raw_text=raw_text,
        postamble=postamble
    ).to_messages()

    model = ChatOpenAI()
    print("Querying LLM...")
    result = model(request, temperature=0)
    print("Model response:")
    print(result.content)

    return result.content


def process_pdf_files(file_list):
    for file_path in file_list:
        raw_text = extract_text(file_path)
        print(f"Extracted text for {file_path}:\n{raw_text}")
        extracted_json = extract_values_from_file(raw_text)

        json_path = f"{file_path}.json"
        with open(json_path, "w") as f:
            f.write(extracted_json)


def main():
    load_dotenv()

    if len(sys.argv) < 2:
        show_usage_and_exit()

    input_path = sys.argv[1]
    print(f"Processing path: {input_path}")

    file_list = enumerate_pdf_files(input_path)
    print(f"Found {len(file_list)} PDF(s) to process.")

    process_pdf_files(file_list)


if __name__ == '__main__':
    main()
