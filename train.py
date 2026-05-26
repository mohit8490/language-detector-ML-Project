import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')
 

# step 1: Load dataset
df = pd.read_csv(r"Machine Learning\ML Project\Supervised ML Project\Intermediate\Language Detector\data\language.csv")

print(df.columns)


# stop words is not useful in language detection because stop words are common words that appear in many languages and can be present in the text data regardless of the language being detected. Removing stop words can lead to loss of important information that is crucial for accurate language detection. Therefore, it is generally recommended to keep stop words in the text data when performing language detection tasks.
# stop_words = set(stopwords.words('english')) 

# step 2: clean the text data by removing special characters, numbers, punctuations etc.
def clean_text(Text):

    # lowercase
    text = Text.lower()

    # remove urls
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # remove punctuation.  not required for language detection because punctuation can be an important feature for distinguishing between different languages. Different languages may have unique punctuation marks or patterns that can help in identifying the language of the text. Removing punctuation may lead to loss of important information that is crucial for accurate language detection. Therefore, it is generally recommended to keep punctuation in the text data when performing language detection tasks.
    # text = text.translate(str.maketrans('', '', string.punctuation))

    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenize.  not required for language detection because we are using character-level n-grams in TF-IDF vectorization, which captures the character patterns and sequences that are indicative of different languages. Tokenization is typically used in word-level analysis, but for language detection, character-level features are more effective in capturing the unique characteristics of each language. Therefore, we can skip tokenization and directly apply TF-IDF vectorization on the cleaned text data for language detection tasks.
    # words = word_tokenize(text)

    # # remove numbers. not required for language detection because numbers can be an important feature for distinguishing between different languages. Different languages may have unique number systems or patterns that can help in identifying the language of the text. Removing numbers may lead to loss of important information that is crucial for accurate language detection. Therefore, it is generally recommended to keep numbers in the text data when performing language detection tasks.
    # words = [word for word in words if not word.isdigit()]

    return text

# step 3: Apply cleaning  ( apply this clean_text function to the 'text' column of the dataframe and create a new column 'clean_text' to store the cleaned text data.)
df['clean_text'] = df['Text'].apply(clean_text)




# step 4: split the dataset into training and testing sets.
X = df['clean_text']  # features (cleaned text data)
y = df['Language']       # target variable (labels)

print(df['Language'].unique())  # check unique languages in the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)




# step 4: convert the text data into numerical features using TF-IDF vectorization.
vectorizer = TfidfVectorizer(
    analyzer     = 'char',   # character-level n-grams
    ngram_range  = (1, 3),      # unigrams, bigrams, trigrams
    max_features = 50000,
    strip_accents= None,        # NEVER strip accents for language detection
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# step 5: select best model based on performance metrics (accuracy, precision, recall, F1-score) and save the trained model and the vectorizer for future use.
# *********************************************************************
# select Model which has best performance on the test set based on accuracy, precision, recall and F1-score. You can use any of the following models for this task: Logistic Regression, Decision Tree, Naive Bayes, SVM, KNN etc.
# *********************************************************************

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Multinomial Naive Bayes': MultinomialNB(),
    'LinearSVC': LinearSVC(max_iter=5000),
    'KNN': KNeighborsClassifier()
}

result = []

for name, model in models.items():

    # train model
    model.fit(X_train_tfidf, y_train)

    # prediction
    y_pred = model.predict(X_test_tfidf)

    # metrics
    acc = accuracy_score(y_test, y_pred)

    pre = precision_score(
        y_test,
        y_pred,
        average='weighted'
    )

    rec = recall_score(
        y_test,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average='weighted'
    )

    # save result
    result.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(pre, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4)
    })



# Step 6: Save best model and vectorizer
result_df = pd.DataFrame(result)
best_name = result_df.sort_values(by='Accuracy', ascending=False).iloc[0]['Model']
best_model = models[best_name]
print(f"\nBest model: {best_name}")





# step 8: save the trained model and the vectorizer for future use.
joblib.dump(best_model, r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Language Detector\models\language_model.pkl')
joblib.dump(vectorizer, r'D:\AIML\Machine Learning\ML Project\Supervised ML Project\Intermediate\Language Detector\models\vectorizer.pkl')

print("Model saved successfully.")