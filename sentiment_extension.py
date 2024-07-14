from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib
import pandas as pd
import re
import string
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for your Flask app

# Load the model and vectorizer
lr = joblib.load('sentiment_model.pkl')
vectorization = joblib.load('vectorizer.pkl')

# Preprocessing for model input
def wp(text):
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to predict sentiment
def output_label(n):
    if n == 0:
        return "Negative"
    elif n == 1:
        return "Neutral"
    elif n == 2:
        return "Positive"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wp)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_lr = lr.predict(new_xv_test)
    return output_label(pred_lr[0])

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    feedback = data['feedback']
    url = data['url']
    sentiment = manual_testing(feedback)

    if sentiment == "Positive":
        message = "Great! Glad you liked the page."
        category = 'liked_pages'
    elif sentiment == "Negative":
        message = "Sorry for your inconvenience, we'll try to improve the page."
        category = 'unliked_pages'
    else:
        message = "Thanks for your feedback."
        category = 'neutral_pages'

    # Save URL based on category (You mentioned saving the link only)
    save_url(url, category)

    return jsonify({"message": message})

def save_url(url, category):
    if not os.path.exists(category):
        os.makedirs(category)
    with open(os.path.join(category, 'urls.txt'), 'a') as f:
        f.write(f"{url}\n")

if __name__ == '__main__':
    app.run(debug=True)
