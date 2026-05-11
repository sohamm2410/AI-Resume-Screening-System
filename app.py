import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Skills list
skills = [
    "python",
    "sql",
    "machine learning",
    "data analysis",
    "power bi",
    "excel",
    "tableau",
    "deep learning"
]

# Function to extract text from PDF
def extract_text(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""

    for page in pdf_reader.pages:
        extracted = page.extract_text()

        if extracted:
            text += extracted

    return text


# Streamlit UI
st.title("📄 AI Resume Screening System")

st.write("Upload your resume and compare it with a job description.")

# Upload PDF
uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf"]
)

# Job description input
job_description = st.text_area(
    "Paste Job Description"
)

# Main Logic
if uploaded_resume and job_description:

    # Extract resume text
    resume_text = extract_text(uploaded_resume)

    # Success message
    st.success("✅ Resume Uploaded Successfully")

    # Convert to lowercase
    resume_lower = resume_text.lower()

    # TF-IDF comparison
    text_data = [resume_text, job_description]

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(text_data)

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )

    # Match score
    match_score = similarity[0][0] * 100

    st.subheader(f"📊 Match Score: {match_score:.2f}%")

    # Progress bar
    st.progress(int(match_score))

    # Skill Extraction
    found_skills = []

    for skill in skills:
        if skill in resume_lower:
            found_skills.append(skill)

    st.subheader("✅ Skills Found")

    if found_skills:
        st.write(found_skills)
    else:
        st.write("No matching skills found.")

    # Missing Skills
    missing_skills = []

    for skill in skills:
        if skill not in resume_lower:
            missing_skills.append(skill)

    st.subheader("❌ Missing Skills")

    if missing_skills:
        st.write(missing_skills)
    else:
        st.write("No missing skills.")

    # Resume Preview
    st.subheader("📄 Resume Text Preview")

    st.write(resume_text[:1000])