import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# ===============================
# 3. INPUT TEXT
# ===============================
text = "The children are running faster and better than the others"

# ===============================
# 4. TOKENIZATION
# ===============================
tokens = word_tokenize(text)

# ===============================
# 5. STEMMING
# ===============================
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens]

# ===============================
# 6. OUTPUT
# ===============================
print("Original Tokens:")
print(tokens)

print("\nStemmed Tokens:")
print(stemmed_words)