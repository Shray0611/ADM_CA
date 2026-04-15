import os
import io
import pdfplumber
import docx
from typing import List, Dict, Any
from utils.text_cleaner import clean_text

def process_uploaded_files(uploaded_files) -> List[Dict[str, Any]]:
    """
    Takes a list of file-like objects (from Streamlit),
    parses them according to their extension, and returns structured data.
    """
    documents = []

    for file in uploaded_files:
        filename = file.name
        raw_text = ""
        
        try:
            if filename.endswith(".txt"):
                raw_text = file.getvalue().decode("utf-8")
            
            elif filename.endswith(".pdf"):
                with pdfplumber.open(io.BytesIO(file.getvalue())) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            raw_text += page_text + "\n"
            
            elif filename.endswith(".docx"):
                doc = docx.Document(io.BytesIO(file.getvalue()))
                raw_text = "\n".join([para.text for para in doc.paragraphs])
            
            else:
                # Unsupported format
                continue
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        processed_text = clean_text(raw_text)

        # Skip empty documents
        if len(processed_text.strip()) == 0:
            continue

        documents.append({
            "filename": filename,
            "raw_text": raw_text,
            "processed_text": processed_text
        })
        
    return documents
