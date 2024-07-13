import pandas as pd
import glob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import re
from bs4 import BeautifulSoup
from contractions import contractions_dict
from nltk.tokenize import word_tokenize

path_0 = 'D:/python/review_polarity/txt_sentoken/pos/'
path_1 = 'D:/python/review_polarity/txt_sentoken/neg/'

txt_files_0 = glob.glob(path_0 + '*.txt')
txt_files_1 = glob.glob(path_1 + '*.txt')

data = []
for file in txt_files_0:
    with open(file, 'r') as f:
        content = f.read()
        data.append((content, 0))

for file in txt_files_1:
    with open(file, 'r') as f:
        content = f.read()
        data.append((content, 1))

data = pd.DataFrame(data, columns=['Text', 'Label'])

# Preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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
    
    # Remove punctuation and non-alphabetic characters
    tokens = [word.lower() for word in tokens if word.isalpha()]
    
    # Define custom list of stopwords excluding 
    custom_stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in custom_stop_words]
    
    # Lemmatize the words
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

data['ProcessedText'] = data['Text'].apply(preprocess_text)

# Featurization - Bag of Words (with n-grams)
ngram_range = (1, 2)  
max_features = 5000  

bow_vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
X_bow = bow_vectorizer.fit_transform(data['ProcessedText'])
y = data['Label']

# Featurization - TF-IDF
tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
X_tfidf = tfidf_vectorizer.fit_transform(data['ProcessedText'])

# Splitting data into train and test sets
X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.15, random_state=164)
X_train_tfidf, X_test_tfidf, y_train2, y_test2 = train_test_split(X_tfidf, y, test_size=0.15, random_state=164)

# Modeling - Logistic Regression
model_lr_bow = LogisticRegression(max_iter=10000)  # Increased max_iter for better convergence
model_lr_tfidf = LogisticRegression(max_iter=10000)

param_grid_lr = {'C': [0.1, 1, 10]}  # Regularization parameter

grid_lr_bow = GridSearchCV(model_lr_bow, param_grid_lr)
grid_lr_tfidf = GridSearchCV(model_lr_tfidf, param_grid_lr)

grid_lr_bow.fit(X_train_bow, y_train)
grid_lr_tfidf.fit(X_train_tfidf, y_train2)

model_lr_bow = grid_lr_bow.best_estimator_
model_lr_tfidf = grid_lr_tfidf.best_estimator_

# Predictions
y_pred_lr_bow = model_lr_bow.predict(X_test_bow)
y_pred_lr_tfidf = model_lr_tfidf.predict(X_test_tfidf)

# Accuracy and Confusion Matrix - Logistic Regression - Bag of Words
accuracy_lr_bow = accuracy_score(y_test, y_pred_lr_bow)
conf_matrix_lr_bow = confusion_matrix(y_test, y_pred_lr_bow)

print("Logistic Regression - Bag of Words")
print("Accuracy:", accuracy_lr_bow)
print("Confusion Matrix:")
print(conf_matrix_lr_bow)

# Accuracy and Confusion Matrix - Logistic Regression - TF-IDF
accuracy_lr_tfidf = accuracy_score(y_test2, y_pred_lr_tfidf)
conf_matrix_lr_tfidf = confusion_matrix(y_test2, y_pred_lr_tfidf)

print("\nLogistic Regression - TF-IDF")
print("Accuracy:", accuracy_lr_tfidf)
print("Confusion Matrix:")
print(conf_matrix_lr_tfidf)

# Modeling - Support Vector Machine (SVM)
model_svm_bow = SVC()
model_svm_tfidf = SVC()

param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}  # Tuning SVM hyperparameters

grid_svm_bow = GridSearchCV(model_svm_bow, param_grid_svm)
grid_svm_tfidf = GridSearchCV(model_svm_tfidf, param_grid_svm)

grid_svm_bow.fit(X_train_bow, y_train)
grid_svm_tfidf.fit(X_train_tfidf, y_train2)

model_svm_bow = grid_svm_bow.best_estimator_
model_svm_tfidf = grid_svm_tfidf.best_estimator_

# Predictions
y_pred_svm_bow = model_svm_bow.predict(X_test_bow)
y_pred_svm_tfidf = model_svm_tfidf.predict(X_test_tfidf)

# Accuracy and Confusion Matrix - SVM - Bag of Words
accuracy_svm_bow = accuracy_score(y_test, y_pred_svm_bow)
conf_matrix_svm_bow = confusion_matrix(y_test, y_pred_svm_bow)

print("\nSupport Vector Machine - Bag of Words")
print("Accuracy:", accuracy_svm_bow)
print("Confusion Matrix:")
print(conf_matrix_svm_bow)

# Accuracy and Confusion Matrix - SVM - TF-IDF
accuracy_svm_tfidf = accuracy_score(y_test2, y_pred_svm_tfidf)
conf_matrix_svm_tfidf = confusion_matrix(y_test2, y_pred_svm_tfidf)

print("\nSupport Vector Machine - TF-IDF")
print("Accuracy:", accuracy_svm_tfidf)
print("Confusion Matrix:")
print(conf_matrix_svm_tfidf)

# Save the trained models
joblib.dump(model_lr_bow, 'D:/python/sentiment_model_lr_bow.pkl')
joblib.dump(model_lr_tfidf, 'D:/python/sentiment_model_lr_tfidf.pkl')
joblib.dump(model_svm_bow, 'D:/python/sentiment_model_svm_bow.pkl')
joblib.dump(model_svm_tfidf, 'D:/python/sentiment_model_svm_tfidf.pkl')
joblib.dump(tfidf_vectorizer, 'D:/python/tfidf_vectorizer.pkl')

# Initialize a dictionary to store results
results = {}

# Logistic Regression - Bag of Words
results['Logistic Regression - Bag of Words'] = {'Accuracy': accuracy_lr_bow, 'Confusion Matrix': conf_matrix_lr_bow}

# Logistic Regression - TF-IDF
results['Logistic Regression - TF-IDF'] = {'Accuracy': accuracy_lr_tfidf, 'Confusion Matrix': conf_matrix_lr_tfidf}

# Support Vector Machine - Bag of Words
results['Support Vector Machine - Bag of Words'] = {'Accuracy': accuracy_svm_bow, 'Confusion Matrix': conf_matrix_svm_bow}

# Support Vector Machine - TF-IDF
results['Support Vector Machine - TF-IDF'] = {'Accuracy': accuracy_svm_tfidf, 'Confusion Matrix': conf_matrix_svm_tfidf}

# Print the results
print("\nModel\t\t\t\t\t\tAccuracy\tConfusion Matrix")
print("-" * 80)
for model, result in results.items():
    accuracy = result['Accuracy']
    conf_matrix = result['Confusion Matrix']
    print(f"{model}\t\t{accuracy:.4f}\t\t{conf_matrix[0][0]:<4} {conf_matrix[0][1]:<4}")
    print("\t\t\t\t\t\t\t\t\t\t", f"{conf_matrix[1][0]:<4} {conf_matrix[1][1]:<4}")
    print("-" * 80)

with open('results.txt', 'w') as f:
    f.write("Model\t\t\t\t\t\tAccuracy\tConfusion Matrix\n")
    f.write("-" * 80 + "\n")
    for model, result in results.items():
        accuracy = result['Accuracy']
        conf_matrix = result['Confusion Matrix']
        f.write(f"{model}\t\t{accuracy:.4f}\t\t{conf_matrix[0][0]:<4} {conf_matrix[0][1]:<4}\n")
        f.write("\t\t\t\t\t\t\t\t\t\t\t" + f"{conf_matrix[1][0]:<4} {conf_matrix[1][1]:<4}\n")
        f.write("-" * 80 + "\n")

print("Results saved in 'results.txt'")
