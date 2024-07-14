import joblib
import pandas as pd
import re
import string

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

# Test with user input
text = input("Enter text for sentiment analysis: ")
print(manual_testing(text))
