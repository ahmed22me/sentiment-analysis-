

import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from bs4 import BeautifulSoup
from contractions import contractions_dict
from nltk.tokenize import word_tokenize
import glob

# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # Convert contractions
    for word in text.split():
        if word.lower() in contractions_dict:
            text = text.replace(word, contractions_dict[word.lower()])
    
    # Remove special characters, punctuation, and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Load the trained models and vectorizer
model_lr_tfidf = joblib.load('D:/python/sentiment_model_lr_tfidf.pkl')
model_svm_tfidf = joblib.load('D:/python/sentiment_model_svm_tfidf.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Test Data Paths
test_path_0 = 'D:/python/mix20_rand700_tokens_cleaned/tokens/pos/'
test_path_1 = 'D:/python/mix20_rand700_tokens_cleaned/tokens/neg/'

# Load test data
def load_test_data(test_path, label):
    txt_files = glob.glob(test_path + '*.txt')
    data = []
    for file in txt_files:
        with open(file, 'r') as f:
            content = f.read()
            data.append((content, label))
    return pd.DataFrame(data, columns=['Text', 'Label'])

# Load test data
test_data_0 = load_test_data(test_path_0, 0)
test_data_1 = load_test_data(test_path_1, 1)

# Preprocess test data
test_data_0['ProcessedText'] = test_data_0['Text'].apply(preprocess_text)
test_data_1['ProcessedText'] = test_data_1['Text'].apply(preprocess_text)

# Filter out empty processed texts (if any)
test_data_0 = test_data_0[test_data_0['ProcessedText'].apply(lambda x: len(x) > 0)]
test_data_1 = test_data_1[test_data_1['ProcessedText'].apply(lambda x: len(x) > 0)]

# Transform test data using TF-IDF vectorizer
X_test_0 = tfidf_vectorizer.transform(test_data_0['ProcessedText'])
X_test_1 = tfidf_vectorizer.transform(test_data_1['ProcessedText'])

# Predictions using Logistic Regression models
y_pred_lr_tfidf_0 = model_lr_tfidf.predict(X_test_0)
y_pred_lr_tfidf_1 = model_lr_tfidf.predict(X_test_1)

# Predictions using SVM models
y_pred_svm_tfidf_0 = model_svm_tfidf.predict(X_test_0)
y_pred_svm_tfidf_1 = model_svm_tfidf.predict(X_test_1)

# Evaluate Logistic Regression models
accuracy_lr_tfidf_0 = accuracy_score(test_data_0['Label'], y_pred_lr_tfidf_0)
conf_matrix_lr_tfidf_0 = confusion_matrix(test_data_0['Label'], y_pred_lr_tfidf_0)

accuracy_lr_tfidf_1 = accuracy_score(test_data_1['Label'], y_pred_lr_tfidf_1)
conf_matrix_lr_tfidf_1 = confusion_matrix(test_data_1['Label'], y_pred_lr_tfidf_1)



# Evaluate SVM models
accuracy_svm_tfidf_0 = accuracy_score(test_data_0['Label'], y_pred_svm_tfidf_0)
conf_matrix_svm_tfidf_0 = confusion_matrix(test_data_0['Label'], y_pred_svm_tfidf_0)

accuracy_svm_tfidf_1 = accuracy_score(test_data_1['Label'], y_pred_svm_tfidf_1)
conf_matrix_svm_tfidf_1 = confusion_matrix(test_data_1['Label'], y_pred_svm_tfidf_1)

# Print results
print("\nLogistic Regression - TF-IDF - Positive Sentiment")
print("Accuracy:", accuracy_lr_tfidf_0)
print("Confusion Matrix:")
print(conf_matrix_lr_tfidf_0)

print("\nSVM - TF-IDF - Positive Sentiment")
print("Accuracy:", accuracy_svm_tfidf_0)
print("Confusion Matrix:")
print(conf_matrix_svm_tfidf_0)

print("\nLogistic Regression - TF-IDF - Negative Sentiment")
print("Accuracy:", accuracy_lr_tfidf_1)
print("Confusion Matrix:")
print(conf_matrix_lr_tfidf_1)

print("\nSVM - TF-IDF - Negative Sentiment")
print("Accuracy:", accuracy_svm_tfidf_1)
print("Confusion Matrix:")
print(conf_matrix_svm_tfidf_1)




output_file_path = 'D:/python/test_Script_output.txt'


with open(output_file_path, 'w') as f:
    f.write("Logistic Regression - TF-IDF - Positive Sentiment\n")
    f.write(f"Accuracy: {accuracy_lr_tfidf_0}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{conf_matrix_lr_tfidf_0}\n\n")

    f.write("Logistic Regression - TF-IDF - Negative Sentiment\n")
    f.write(f"Accuracy: {accuracy_lr_tfidf_1}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{conf_matrix_lr_tfidf_1}\n\n")

    f.write("SVM - TF-IDF - Positive Sentiment\n")
    f.write(f"Accuracy: {accuracy_svm_tfidf_0}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{conf_matrix_svm_tfidf_0}\n\n")
    
    f.write("SVM - TF-IDF - Negative Sentiment\n")
    f.write(f"Accuracy: {accuracy_svm_tfidf_1}\n")
    f.write("Confusion Matrix:\n")
    f.write(f"{conf_matrix_svm_tfidf_1}\n")

