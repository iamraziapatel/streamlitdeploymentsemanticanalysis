import streamlit as st
import joblib
import re
import string
import spacy

# Load the spaCy model
# Provide the absolute path to the 'en_core_web_sm' model data directory
model_path = "en_core_web_sm"

nlp = spacy.load(model_path)

# Load the trained XGBoost model and label encoder
model = joblib.load('xgboost_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
# Load the TfidfVectorizer (or other transformer)
tfidfvect = joblib.load('tfidfvect.pkl')


# Define the preprocess function using spaCy tokenizer
def preprocess_text(text):
    # Remove @mentions, URLs, and hashtags
    text = re.sub(r'@+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    text = re.sub(r'#', '', text)

    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize using spaCy
    tokens = [token.text for token in nlp(text)]

    # Convert tokens to lowercase and filter out non-empty tokens
    cleaned_tokens = [token.lower() for token in tokens if token.strip()]

    # Lemmatize using spaCy
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(cleaned_tokens))]

    # Remove stop words
    cleaned_tokens = [token for token in lemmatized_tokens if token not in nlp.Defaults.stop_words]
    # Join the preprocessed tokens into a single string
    preprocessed_text = ' '.join(cleaned_tokens)

    # Return the preprocessed text as a single string
    return preprocessed_text


def main():
    st.title("Semantic Analysis of tweets")

    # Create a text input widget
    user_tweet = st.text_area("Enter text for analysis", "")

    # Check if the user submitted input
    if st.button("Analyze"):
        try:
            # Preprocess the input tweet using the loaded preprocessing function (if needed)
            preprocessed_tweet = preprocess_text(user_tweet)

            # Transform the preprocessed text using the loaded TfidfVectorizer
            tfidf_features = tfidfvect.transform([preprocessed_tweet])

            # Make predictions using the loaded XGBoost model
            prediction = model.predict(tfidf_features)[0]

            # Decode the predicted class label
            predicted_class = label_encoder.inverse_transform([prediction])[0]

            # Display the prediction to the user
            st.write("Analysis results:")
            st.write(f"Predicted Class: {predicted_class}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()

