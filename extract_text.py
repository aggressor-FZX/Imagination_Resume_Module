import os
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def main():
    """Extracts text from resumes and job ad."""
    test_dir = 'test'
    analyst_programmer_pdf = os.path.join(test_dir, 'AnalystProgrammer.pdf')
    dogwood_pdf = os.path.join(test_dir, 'dogwood.pdf')
    job_ad_path = os.path.join(test_dir, 'JOB_ad')

    # Extract text from PDFs
    analyst_programmer_text = extract_text_from_pdf(analyst_programmer_pdf)
    dogwood_text = extract_text_from_pdf(dogwood_pdf)

    # Read job ad text
    with open(job_ad_path, 'r') as f:
        job_ad_text = f.read()

    # Save extracted text to files
    with open(os.path.join(test_dir, 'analyst_programmer_resume.txt'), 'w') as f:
        f.write(analyst_programmer_text)
    with open(os.path.join(test_dir, 'dogwood_resume.txt'), 'w') as f:
        f.write(dogwood_text)
    with open(os.path.join(test_dir, 'job_ad.txt'), 'w') as f:
        f.write(job_ad_text)

if __name__ == '__main__':
    main()
