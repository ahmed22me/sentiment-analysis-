
# import requests
# from urllib.parse import urlparse
# from bs4 import BeautifulSoup
# import pandas as pd
# import joblib
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Load the saved models and vectorizer
# model_lr_bow = joblib.load('D:/python/sentiment_model_lr_bow.pkl')
# model_lr_tfidf = joblib.load('D:/python/sentiment_model_lr_tfidf.pkl')
# model_svm_bow = joblib.load('D:/python/sentiment_model_svm_bow.pkl')
# model_svm_tfidf = joblib.load('D:/python/sentiment_model_svm_tfidf.pkl')
# tfidf_vectorizer = joblib.load('D:/python/tfidf_vectorizer.pkl')


# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# def preprocess_text(text):
#     # Text preprocessing steps
#     text = BeautifulSoup(text, "html.parser").get_text()
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
#     tokens = [token for token in tokens if token not in stop_words]
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(token) for token in tokens]
#     cleaned_text = ' '.join(tokens)
#     return cleaned_text

# def predict_sentiment(text):
#     processed_text = preprocess_text(text)
#     processed_text_vectorized = tfidf_vectorizer.transform([processed_text])
#     prediction = model_lr_tfidf.predict(processed_text_vectorized)
#     sentiment = 'Negative' if prediction == 1 else 'Positive'
#     return sentiment

# def extract_text_from_url(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     text = soup.get_text()
#     return text

# def analyze_web_page(url):
#     text = extract_text_from_url(url)
#     sentiment = predict_sentiment(text)
#     return sentiment



# url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7380170/'
# predicted_sentiment = analyze_web_page(url)
# print("Predicted Sentiment for", url, "is:", predicted_sentiment)



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
    
    # Predict sentiment using each model
    predictions = {
        'lr_bow': model_lr_bow.predict(processed_text_vectorized),
        'lr_tfidf': model_lr_tfidf.predict(processed_text_vectorized),
        'svm_bow': model_svm_bow.predict(processed_text_vectorized),
        'svm_tfidf': model_svm_tfidf.predict(processed_text_vectorized)
    }
    
    # Map predictions to sentiment labels
    sentiments = {}
    for model_name, prediction in predictions.items():
        if prediction == 0:
            sentiments[model_name] = 'Positive'
        else:
            sentiments[model_name] = 'Negative'
    
    return sentiments

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    return text

def analyze_web_page(url):
    text = extract_text_from_url(url)
    sentiments = predict_sentiment(text)
    return sentiments

# sample_text = "I really enjoyed this movie. The acting was superb!"
sample_text = "I really hate this movie. The acting was rood!"

predicted_simple_sentiment = predict_sentiment(sample_text)
print("Sample Text:", sample_text)
print("Predicted Sentiment using 4 Models:", predicted_simple_sentiment)

url = 'https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7380170/'
predicted_sentiments = analyze_web_page(url)
print("Predicted Sentiments for", url, "using 4 Models:")
for model_name, sentiment in predicted_sentiments.items():
    print(f"{model_name}: {sentiment}")

