from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import shap

nltk.download('stopwords')
nltk.download('punkt')

train_data = pd.read_csv('multilabel_train.csv')
test_data = pd.read_csv('multilabel_test.csv')

import demoji

def preprocess_text(text):
    text = demoji.replace(text, '')
    text = text.lower()
    text = re.sub(r'[^\w\s#]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(filtered_tokens)
    return text

train_data['preprocessed_text'] = train_data['Description'].apply(preprocess_text)

X_train = train_data['preprocessed_text']
y_train = train_data[['Commenting', 'Ogling/Facial Expressions/Staring', 'Touching /Groping']]

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

classifier = MultiOutputClassifier(LogisticRegression())
classifier.fit(X_train_tfidf, y_train)

base_estimator = classifier.estimators_[0]

explainer = shap.Explainer(base_estimator, X_train_tfidf)

def classify_text_with_shap(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    shap_values = explainer(text_tfidf)
    shap_summary = np.mean(np.abs(shap_values.values), axis=0)
    top_features = np.argsort(shap_summary)[::-1][:5]  # Top 5 features
    top_features_names = tfidf_vectorizer.get_feature_names_out()[top_features]
    return top_features_names.tolist()  # Convert to list

def classify_text(text):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction = classifier.predict(text_tfidf)
    predicted_indices = np.where(prediction[0] == 1)[0]
    predicted_categories = train_data.columns[1:][predicted_indices]
    return predicted_categories.tolist()  # Convert to list

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ""
    predicted_categories = []
    top_features = []
    if request.method == 'POST':
        input_text = request.form['input_text']
        if input_text:
            predicted_categories = classify_text(input_text)
            top_features = classify_text_with_shap(input_text)
    return render_template('index.html', input_text=input_text, predicted_categories=predicted_categories, top_features=top_features)

if __name__ == '__main__':
    app.run(debug=True)
