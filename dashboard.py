import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load your dataset and preprocess 'Label' as you mentioned before
df = pd.read_csv('data.csv')
df['Label'] = df['Label'].map({"FAKE": 0, "REAL": 1})
df = df.drop('ID', axis=1)

# Assuming your dataset has 'Text' and 'Label' columns
X = df['Text'].values
y = df['Label'].values

# Vectorize the text data using TF-IDF from scikit-learn
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Create and fit the Logistic Regression model
logreg_classifier = LogisticRegression(random_state=17)
logreg_classifier.fit(X_tfidf, y)

# Function to predict if a text is real or fake
def predict_text(label_text):
    # Vectorize the input text using the same vectorizer
    text_tfidf = vectorizer.transform([label_text])

    # Make a prediction
    prediction = logreg_classifier.predict(text_tfidf)

    # Return the prediction
    return "REAL" if prediction[0] == 1 else "FAKE"

# Streamlit app
def main():
    st.title("Fake News Detection")

    # User input text
    user_input = st.text_area("Enter a text:", "This is a sample text.")

    # Make prediction on button click
    if st.button("Predict"):
        result = predict_text(user_input)
        st.success(f"The model predicts: {result}")

if __name__ == "__main__":
    main()
