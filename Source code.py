!pip install gradio
!pip install gradio --quiet
import nltk
nltk.download('punkt')
import nltk

# Download the required NLTK data
nltk.download('punkt_tab')

# Load the Dataset
import pandas as pd

# Read the dataset
df = pd.read_csv('twitter_training.csv', header=None, names=['ID', 'Game', 'Sentiment', 'Tweet'])
# Data Exploration
# Display first few rows
print(df.head())

# Shape of the dataset
print("Shape:", df.shape)

# Column names
print("Columns:", df.columns.tolist())
# Data types and non-null values
print(df.info())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Check for duplicates
print("Duplicate rows:", df.duplicated().sum())

# Drop rows with missing values if necessary
df = df.dropna()

# Sentiment Distribution
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the distribution of sentiments
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=df)
plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
# Preprocessing Text Data
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions and hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)
df['Cleaned_Tweet'] = df['Tweet'].apply(preprocess_text)

# Feature Extraction (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Cleaned_Tweet'])
y = df['Sentiment']

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Predict Sentiment for New Tweets
def predict_sentiment(tweet):
    cleaned_tweet = preprocess_text(tweet)
    tweet_vector = tfidf.transform([cleaned_tweet])
    prediction = model.predict(tweet_vector)
    return prediction[0]

# Example
new_tweet = "I love Borderlands! The gameplay is amazing."
print("Predicted Sentiment:", predict_sentiment(new_tweet))

# Interactive App (Optional)
import gradio as gr

def sentiment_analysis(tweet):
    prediction = predict_sentiment(tweet)
    return f"Predicted Sentiment: {prediction}"

iface = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(lines=2, placeholder="Enter a tweet about Borderlands..."),
    outputs="text",
    title="Borderlands Tweet Sentiment Analysis",
    description="Enter a tweet to predict its sentiment (Positive, Negative, Neutral, Irrelevant)."
)

iface.launch()
