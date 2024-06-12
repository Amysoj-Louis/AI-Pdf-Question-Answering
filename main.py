import os
import fitz
from PIL import Image, ImageEnhance
import pytesseract
import google.generativeai as genai
import logging
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Tesseract command
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# Retrieve the API key from environment variable
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError(
        "No API key found in environment variables. Please set the 'GEMINI_API_KEY' variable.")

# Configure the Generative AI API
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Define the Generative Model
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="""
    **Input:** Plain text extracted from an image, potentially containing questions.
    **Task:** 
    1. **Question Identification:** Identify any questions present in the text.
    2. **Information Extraction:** Extract key information related to the question from the surrounding text. This may include entities, concepts, and relationships.
    3. **Answer Generation:**
        * For MCQs:
            * If options are explicitly numbered or lettered, identify the most likely answer based on extracted information and reasoning. Provide the answer in the format "Option [Letter]: [Text]".
            * If options are not explicitly provided, generate a shortlist of the most plausible answers based on the extracted information and reasoning. 
        * For Open-ended questions: 
            * If the question uses "or", generate multiple possible answers that address each alternative presented.
            * If the question uses "based on", analyze the extracted information and use reasoning to formulate an answer that best explains the concept or relationship being queried.
            * When a word limit is specified, strive for concise and informative answers within the limit.
            * If no word limit exists, prioritize providing a clear and comprehensive answer.
        * For Any Other questions: 
            * If the question uses "or", generate multiple possible answers that address each alternative presented.
            * If the question uses "based on", analyze the extracted information and use reasoning to formulate an answer that best explains the concept or relationship being queried.
            * When a word limit is specified, strive for concise and informative answers within the limit.
            * If no word limit exists, prioritize providing a clear and comprehensive answer.
    **Output:** 
    * For MCQs: Option [Letter]: [Text] (or list of shortlisted options)
    * For Open Ended Questions: Answer 1: [Text] (if question uses "or") or explanation of the concept/relationship (if question uses "based on")
    **Important:** 
    * Focus on providing the most relevant and informative answer(s) without unnecessary explanations.
    """
)


def ensure_directories_exist(*dirs: str):
    """Ensure the provided directories exist, creating them if necessary."""
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def process_pdf(pdf_path: str) -> Tuple[List[str], List[str]]:
    """Extracts text and generates answers from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"Failed to open PDF file {pdf_path}: {e}")
        return [], []

    text_outputs = []
    answers = []

    ensure_directories_exist("default_image", "pro_image")

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, doc, page_num)
                   for page_num in range(len(doc))]
        for future in futures:
            text, answer = future.result()
            text_outputs.append(text)
            answers.append(answer)

    return text_outputs, answers


def process_page(doc, page_num: int) -> Tuple[str, str]:
    """Processes a single page: converts to image, extracts text, and generates an answer."""
    try:
        page = doc.load_page(page_num)
        image_path, processed_image_path = save_page_as_image(page, page_num)
        extracted_text = extract_text_from_image(processed_image_path)
        answer = generate_answer(extracted_text)
        logging.info(f"Processed Page {page_num + 1}")
        return extracted_text, answer
    except Exception as e:
        logging.error(f"Error processing page {page_num + 1}: {e}")
        return "", ""


def save_page_as_image(page, page_num: int) -> Tuple[str, str]:
    """Save the page as an image and return the image paths."""
    zoom = 3.0  # Reduced zoom for faster processing
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    image_path = f"default_image/page_{page_num + 1}.png"
    pix.save(image_path)

    processed_image_path = enhance_image_contrast(image_path, page_num)
    return image_path, processed_image_path


def enhance_image_contrast(image_path: str, page_num: int) -> str:
    """Enhance the contrast of the image and save it."""
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale
    enhancer = ImageEnhance.Contrast(image)
    enhanced_image = enhancer.enhance(2)  # Reduced contrast enhancement
    processed_image_path = f"pro_image/page_{page_num + 1}.png"
    enhanced_image.save(processed_image_path)
    return processed_image_path


def extract_text_from_image(image_path: str) -> str:
    """Extract text from the image using Pytesseract."""
    try:
        return pytesseract.image_to_string(image_path, lang='eng', config='--psm 6')
    except Exception as e:
        logging.error(f"Failed to extract text from image {image_path}: {e}")
        return ""


def generate_answer(extracted_text: str) -> str:
    """Generate an answer for the extracted text using Generative AI."""
    try:
        response = model.generate_content(extracted_text)
        return response.text
    except Exception as e:
        logging.error(f"Failed to generate answer for the text: {e}")
        return "Error in generating answer."


def save_to_file(filename: str, data: List[str], header: str):
    """Save the provided data to a file with a header."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            for idx, content in enumerate(data, start=1):
                file.write(f"{header} {idx}:\n{content}\n\n")
    except Exception as e:
        logging.error(f"Failed to save data to {filename}: {e}")


if __name__ == "__main__":
    pdf_file_path = input("Enter Input Pdf Path: ").strip()

    if not os.path.isfile(pdf_file_path):
        logging.error(f"File {pdf_file_path} does not exist.")
    else:
        text_outputs, answers = process_pdf(pdf_file_path)

        save_to_file('extracted_text.txt', text_outputs, "Text from Page")
        save_to_file('generated_answers.txt', answers, "Answer for Page")

        logging.info("Text and answers saved to TXT files.")
