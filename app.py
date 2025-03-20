import os
import pandas as pd
import spacy
from pdfminer.high_level import extract_text

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Define job description
JOB_DESCRIPTION = "We are looking for a Python developer with experience in machine learning and NLP."
job_doc = nlp(JOB_DESCRIPTION)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    return extract_text(pdf_path)

def calculate_match_score(resume_text, job_doc):
    """Calculate match score based on similarity."""
    resume_doc = nlp(resume_text)
    return resume_doc.similarity(job_doc)

def process_resumes(resume_folder):
    """Process all resumes in a folder and rank them."""
    scores = []
    for filename in os.listdir(resume_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(resume_folder, filename)
            text = extract_text_from_pdf(file_path)
            score = calculate_match_score(text, job_doc)
            scores.append((filename, score))
    
    # Rank resumes by score
    ranked_resumes = sorted(scores, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(ranked_resumes, columns=["Resume", "Score"])

if __name__ == "__main__":
    folder_path = "resumes"  # Folder containing resumes
    results = process_resumes(folder_path)
    print(results)
