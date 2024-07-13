import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import requests


# Load the saved models and vectorizer
model_lr_bow = joblib.load('D:/python/sentiment_model_lr_bow.pkl')
model_lr_tfidf = joblib.load('D:/python/sentiment_model_lr_tfidf.pkl')
model_svm_bow = joblib.load('D:/python/sentiment_model_svm_bow.pkl')
model_svm_tfidf = joblib.load('D:/python/sentiment_model_svm_tfidf.pkl')
tfidf_vectorizer = joblib.load('D:/python/tfidf_vectorizer.pkl')



nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Text preprocessing steps
    text = BeautifulSoup(text, "html.parser").get_text()    
    text = text.lower()    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)    
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def predict_sentiment(text):
    processed_text = preprocess_text(text)
    
    processed_text_vectorized = tfidf_vectorizer.transform([processed_text])
    
    prediction = model_lr_tfidf.predict(processed_text_vectorized)
    
    if prediction == 0:
        sentiment = 'Positive'
    else:
        sentiment = 'Negative'
    
    return sentiment


def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    print("Extracted Text:")
    print(text)
    return text

def analyze_web_page(url):
    text = extract_text_from_url(url)
    sentiment = predict_sentiment(text)
    return sentiment


# sample_text = "I really enjoyed this movie. The acting was superb!"
sample_text = "I really hate this movie. The acting was rood!"

predicted_simple_sentiment = predict_sentiment(sample_text)

print("Sample Text:", sample_text)
print("Predicted Sentiment:", predicted_simple_sentiment)


# url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7380170/' #positive url
url = 'https://www.bbc.com/news/articles/c72p7dj1890o'
predicted_sentiment = analyze_web_page(url)
print("Predicted Sentiment for", url, "is:", predicted_sentiment)


