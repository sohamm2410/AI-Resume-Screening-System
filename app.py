import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Extract text from PDF
def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

# Streamlit UI
st.title("AI Resume Screening System")

uploaded_resume = st.file_uploader("Upload Resume", type=["pdf"])

job_description = st.text_area("Paste Job Description")

if uploaded_resume and job_description:

    resume_text = extract_text(uploaded_resume)

    text_data = [resume_text, job_description]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(text_data)

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])

    match_score = similarity[0][0] * 100

    st.subheader(f"Match Score: {match_score:.2f}%")