import streamlit as st
import joblib
import re

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Language Detector",
    page_icon="🌍",
    layout="centered"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #0f172a, #1e293b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(to right, #2563eb, #38bdf8);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(to right, #1d4ed8, #0ea5e9);
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(
    r"D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Language Detector\models\language_model.pkl"
)

vectorizer = joblib.load(
    r"D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Language Detector\models\vectorizer.pkl"
)

# =====================================================
# CLEAN FUNCTION
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'http\S+|www\S+|https\S+',
        '',
        text
    )

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<h1 class='main-title'>🌍 AI Language Detector</h1>",
    unsafe_allow_html=True
)

# st.markdown(
#     "<p class='sub-title'>Detect languages using NLP & Machine Learning</p>",
#     unsafe_allow_html=True
# )

# =====================================================
# INPUT BOX
# =====================================================

user_input = st.text_area(
    "Enter Text",
    placeholder="Type text here...",
    height=220
)

# =====================================================
# BUTTON
# =====================================================

if st.button("🚀 Detect Language"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        # clean text
        cleaned_text = clean_text(user_input)

        # vectorize
        text_vector = vectorizer.transform([cleaned_text])

        # prediction
        prediction = model.predict(text_vector)[0]

        # =====================================================
        # RESULT
        # =====================================================

        st.success(f"Predicted Language: {prediction}")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("Supported Languages")

languages = [
    "English",
    "Hindi",
    "Tamil",
    "Kannada",
    "Malayalam",
    "French",
    "Spanish",
    "German",
    "Arabic",
    "Russian",
    "Greek",
    "Dutch",
    "Italian",
    "Turkish",
    "Danish",
    "Swedish",
    "Portuguese"
]

for lang in languages:
    st.sidebar.write(f"{lang}")