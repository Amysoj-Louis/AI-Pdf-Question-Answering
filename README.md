# AI-Powered PDF OCR & Question Answering System

A high-performance Python utility designed to extract text from complex PDFs via OCR and automatically generate precise answers using the **Gemini 1.5 Flash** model.

## 📄 Project Overview
This project is an advanced document intelligence system built to handle unstructured data within PDFs. It combines robust OCR capabilities with generative AI to identify questions within scanned documents and provide reasoned, context-aware answers. The system is optimized for speed through multi-threaded parallel processing and is capable of handling diverse question formats.

## ✨ Key Features
- **Intelligent OCR Engine**: Utilizes **Tesseract OCR** and **PyMuPDF** to accurately extract text from complex scanned PDF pages.
- **Context-Aware QA**: Leverages **Google Gemini 1.5 Flash** to identify, extract, and answer questions grounded in the surrounding document context.
- **Multi-Format Question Handling**: Specialized logic for handling MCQs, open-ended questions, and comparative "or" based queries.
- **High-Performance Parallelism**: Implements **ThreadPoolExecutor** for concurrent page processing, significantly reducing analysis time.
- **Image Enhancement**: Integrated support for **Pillow-based** image enhancements to improve OCR accuracy on low-quality scans.

## 🛠️ Tech Stack
- **Language**: Python 3.10
- **AI Engine**: Google Gemini 1.5 Flash
- **OCR Library**: Pytesseract (Tesseract Engine)
- **PDF Processing**: PyMuPDF (fitz)
- **Image Manipulation**: Pillow (PIL)
- **Concurrency**: ThreadPoolExecutor

*“Turning unstructured document scans into actionable knowledge.”*
