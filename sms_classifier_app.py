import streamlit as st
import joblib

# Load the trained model
model = joblib.load('sms_spam_classifier.pkl')
# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the classification function
def classify_sms(message):
    message_tfidf = tfidf_vectorizer.transform([message])
    prediction = model.predict(message_tfidf)
    return "Spam" if prediction[0] == 1 else "Ham"

# Streamlit app
def main():
    st.title("SMS Spam Classification App")
    st.write("Enter your SMS message below:")

    user_input = st.text_input("SMS Message")

    if st.button("Classify"):
        if user_input:
            classification = classify_sms(user_input)
            st.write(f"Classification: {classification}")
        else:
            st.warning("Please enter an SMS message.")

if __name__ == "__main__":
    main()
