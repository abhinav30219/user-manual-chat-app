# User Manual Chat Application

## Overview

The User Manual Chat Application assists users by answering questions about products based on their user manuals. It uses advanced natural language processing techniques to extract relevant information from uploaded PDF files and generates responses through a conversational interface.

## Features

- **PDF Upload**: Users can upload user manuals in PDF format.
- **Content Extraction**: Extracts text, tables, and images from the PDF for processing.
- **AI-Driven Responses**: Utilizes language models to answer questions based on extracted content.
- **Interactive UI**: Built with Streamlit for an easy-to-use interface.

## Prerequisites

- Python 3.8 or higher
- pip and virtualenv

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/abhinav30219/user-manual-chat-app.git
cd user-manual-chat-app

2. Create and Activate a Virtual Environment
For Unix/macOS:

bash
Copy code
python3 -m venv env
source env/bin/activate
For Windows:

cmd
Copy code
python -m venv env
env\Scripts\activate
3. Install Dependencies
Install all required Python packages:

bash
Copy code
pip install -r requirements.txt
Additional Dependencies
Poppler (for PDF processing):
macOS: brew install poppler
Ubuntu: sudo apt-get install -y poppler-utils
Windows: Download Poppler for Windows and add to PATH.
Tesseract OCR (for image to text conversions):
macOS: brew install tesseract
Ubuntu: sudo apt install tesseract-ocr
Windows: Download from Tesseract at UB Mannheim and install. Add the path to the Tesseract executable to your PATH.
4. Run the Application
Start the Streamlit application:

bash
Copy code
streamlit run app.py
This will open the application in your default web browser.

Usage
Uploading a PDF: Click "Upload your user manual in PDF Format" to select and upload a file.
Asking Questions: After uploading the PDF, enter your question in the "Ask questions about your User Manual:" input box and press Enter or the submit button.
Contributing
Contributions to improve the application are welcome. Please follow the guidelines for contributions such as opening issues for discussion before submitting pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Support
For support, contact abhinav4@stanford.edu or open an issue in the GitHub repository.
