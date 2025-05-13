import re
import streamlit as st
import pickle
#import docx
import PyPDF2
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
clf_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    # punctuation = re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_`{|}~]', ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # try using utf-8 encoding
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # if utf-8 fails, try latin-1 encoding
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    """
    Extracts text from uploaded file based on its extension.

    Args:
        uploaded_file: A file object containing the uploaded data.

    Returns:
        Extracted text from the uploaded file.

    Raises:
        ValueError: If the uploaded file has an unsupported extension.
    """

    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file")
    return text


# web app
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload Resume", type=['txt', 'pdf', 'docx'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidf.transform([cleaned_resume])
        prediction_id = clf_model.predict(input_features)[0]
        st.write(prediction_id)

        category_mapping = {
            6: "Data Science",
            12: "HR",
            0: "Advocate",
            1: "Arts",
            24: "Web Designing",
            16: "Mechanical Engineer",
            22: "Sales",
            14: "Health and fitness",
            5: "Civil Engineer",
            15: "Java Developer",
            4: "Business Analyst",
            21: "SAP Developer",
            2: "Automation Testing",
            11: "Electrical Engineering",
            18: "Operations Manager",
            20: "Python Developer",
            8: "DevOps Engineer",
            17: "Network Security Engineer",
            19: "PMO",
            7: "Database",
            13: "Hadoop",
            10: "ETL Developer",
            9: "DotNet Developer",
            3: "Blockchain",
            23: "Testing"
        }

        category_name = category_mapping.get(prediction_id, "unknown")
        st.write("Predicted category:", category_name)
        st.write(prediction_id)


# Python main
if __name__ == "__main__":
    main()
