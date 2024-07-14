import numpy as np
import pandas as pd
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords
import string

# Ignore warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')

# Load the datasets
train_data = pd.read_csv(r'C:\Users\srikr\OneDrive\Desktop\Interview Project\train.csv', encoding='latin1')
test_data = pd.read_csv(r'C:\Users\srikr\OneDrive\Desktop\Interview Project\test.csv', encoding='latin1')

# Combine the datasets
df = pd.concat([train_data, test_data])

# Clean the text data
def remove_unnecessary_characters(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text))
    text = re.sub(r'\s+', ' ', str(text)).strip()
    return text

df['clean_text'] = df['text'].apply(remove_unnecessary_characters)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['normalized_text'] = df['clean_text'].apply(normalize_text)

# Remove stopwords
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text

df['text_without_stopwords'] = df['normalized_text'].apply(remove_stopwords)

# Drop any rows with missing values
df.dropna(inplace=True)

# Encode the sentiment labels
df['sentiment_code'] = df['sentiment'].astype('category').cat.codes

# Preprocessing for model input
def wp(text):
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['selected_text'] = df["text_without_stopwords"].apply(wp)

# Prepare the data for training
X = df['selected_text']
y = df['sentiment_code']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data
vectorization = TfidfVectorizer()
XV_train = vectorization.fit_transform(X_train)
XV_test = vectorization.transform(X_test)

# Train the Logistic Regression model
lr = LogisticRegression(n_jobs=-1)
lr.fit(XV_train, y_train)

# Save the model and vectorizer
joblib.dump(lr, 'sentiment_model.pkl')
joblib.dump(vectorization, 'vectorizer.pkl')

# Evaluate the model
pred_lr = lr.predict(XV_test)
score_lr = accuracy_score(y_test, pred_lr)
print(f'Logistic Regression Accuracy: {score_lr}')

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

# Example usage
text = "I am Sad"
print(manual_testing(text))
