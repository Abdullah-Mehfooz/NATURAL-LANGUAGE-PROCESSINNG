from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "I love machine learning",
    "Machine learning is fun",
    "I love AI"
]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform
X = vectorizer.fit_transform(documents)

# Feature names (words)
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# TF-IDF Matrix
print("\nTF-IDF Matrix:")
print(X.toarray())