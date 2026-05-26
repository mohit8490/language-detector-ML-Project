import joblib
import re

# =====================================================
# STEP 1: Load Saved Model and Vectorizer
# =====================================================

model = joblib.load(
    r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Language Detector\models\language_model.pkl'
)

vectorizer = joblib.load(
    r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Language Detector\models\vectorizer.pkl'
)

# =====================================================
# STEP 2: Clean Text Function
# =====================================================

def clean_text(text):

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =====================================================
# STEP 3: Take User Input
# =====================================================

text = input("Enter text: ")

# clean text
cleaned_text = clean_text(text)

# =====================================================
# STEP 4: Convert Text into TF-IDF Features
# =====================================================

text_vector = vectorizer.transform([cleaned_text])

# =====================================================
# STEP 5: Predict Language
# =====================================================

prediction = model.predict(text_vector)

# =====================================================
# STEP 6: Show Result
# =====================================================

print("\nPredicted Language:", prediction[0])