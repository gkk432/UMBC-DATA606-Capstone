import streamlit as st
import joblib
import gdown
import os

# Function to download files from Google Drive
def download_file_from_google_drive(file_id, output_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)

# Google Drive file IDs
model_file_id = "1SPWyi5617p3SSQ7Zfo3XPpJGiv_DbWLU"  # Logistic Regression Model
vectorizer_file_id = "1xeAhXTkLyIXP0rhZjTaHeeHJeCFSiXf9"  # TF-IDF Vectorizer

# Check and download the model
if not os.path.exists("logistic_regression_model.pkl"):
    download_file_from_google_drive(model_file_id, "logistic_regression_model.pkl")

# Check and download the vectorizer
if not os.path.exists("tfidf_vectorizer.pkl"):
    download_file_from_google_drive(vectorizer_file_id, "tfidf_vectorizer.pkl")

# Load the model and vectorizer
model = joblib.load("logistic_regression_model.pkl")
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit app UI
st.title("Sentiment Analysis APP")
st.subheader("I can classify your review")

# Input from user
user_input = st.text_area("Enter a review:")

# Predict sentiment
if st.button("Classify"):
    if user_input.strip():
        # Transform input and predict
        input_vectorized = tfidf_vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)
        confidence = model.predict_proba(input_vectorized)

        # Display results
        sentiment = "Positive" if prediction[0] == "positive" else "Negative"
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence Level:** {max(confidence[0]) * 100:.2f}%")
    else:
        st.write("Please enter some text to classify.")
