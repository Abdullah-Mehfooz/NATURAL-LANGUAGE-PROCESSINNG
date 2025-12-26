from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
documents = [
    "I love machine learning",
    "Machine learning is fun",
    "I love AI"
]

# Create BoW model
vectorizer = CountVectorizer()

# Fit and transform the data
X = vectorizer.fit_transform(documents)

# Vocabulary
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# BoW Matrix
print("\nBag of Words Matrix:")
print(X.toarray())