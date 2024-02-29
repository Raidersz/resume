import docx

def extract_information_from_docx(docx_path):
    # Load the Word document
    doc = docx.Document(docx_path)

    # Create a string to store extracted information
    extracted_information = ""

    # Iterate through paragraphs and extract text
    for paragraph in doc.paragraphs:
        extracted_information += paragraph.text + "\n"

    return extracted_information

# Example usage
docx_path = r"C:\Users\shrut\Downloads\archive (2)\Resumes/AjayKumar.docx"
resume_text = extract_information_from_docx(docx_path)

# Print the extracted information
print(resume_text)