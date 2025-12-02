from pypdf import PdfReader
import sys

try:
    reader = PdfReader("Proyecto_II_semestre_2025_IA.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    print(text)
except Exception as e:
    print(f"Error reading PDF: {e}")
