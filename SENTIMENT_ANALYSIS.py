
import nltk
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from textblob import TextBlob
from wordcloud import WordCloud

# ===============================
# 3. BASIC TEXTBLOB TEST
# ===============================
text = "The movie was fantastic! I really enjoyed it."
print("Sample Sentiment:", TextBlob(text).sentiment)

# ===============================
# 4. LOAD DATASET
# ===============================
# Make sure CSV file is in same folder
df = pd.read_csv("covid-19_vaccine_tweets_with_sentiment.csv", encoding="latin1")
print(df.head())

# ===============================
# 5. TEXT CLEANING
# ===============================
tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()
en_stop = set(stopwords.words('english'))

def getCleanedText(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in en_stop]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)

df['tweet_text'] = df['tweet_text'].apply(getCleanedText)

# ===============================
# 6. SUBJECTIVITY & POLARITY
# ===============================
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

df['Subjectivity'] = df['tweet_text'].apply(getSubjectivity)
df['Polarity'] = df['tweet_text'].apply(getPolarity)

# ===============================
# 8. SENTIMENT LABELING
# ===============================
def getAnalysis(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

df['Analysis'] = df['Polarity'].apply(getAnalysis)

print(df[['tweet_text','Subjectivity','Polarity','Analysis']].head())

# ===============================
# 9. SCATTER PLOT (Polarity vs Subjectivity)
# ===============================
plt.figure(figsize=(8,6))
plt.scatter(df['Polarity'], df['Subjectivity'], color='green')
plt.title("Sentiment Analysis Scatter Plot")
plt.xlabel("Polarity")
plt.ylabel("Subjectivity")
plt.show()

# ===============================
# 10. SENTIMENT PERCENTAGES
# ===============================
positive = df[df['Analysis'] == 'Positive']
negative = df[df['Analysis'] == 'Negative']
neutral  = df[df['Analysis'] == 'Neutral']

print("Positive tweets:", round((positive.shape[0]/df.shape[0])*100, 2), "%")
print("Negative tweets:", round((negative.shape[0]/df.shape[0])*100, 2), "%")
print("Neutral tweets :", round((neutral.shape[0]/df.shape[0])*100, 2), "%")

# ===============================
# 11. BAR CHART
# ===============================
df['Analysis'].value_counts().plot(kind='bar')
plt.title("Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()