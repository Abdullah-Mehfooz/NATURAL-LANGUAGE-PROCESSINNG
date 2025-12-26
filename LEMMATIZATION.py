import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ===============================
# 3. INPUT TEXT
# ===============================
text = "The children are running faster and better than the others"

# ===============================
# 4. TOKENIZATION
# ===============================
tokens = word_tokenize(text)

# ===============================
# 5. LEMMATIZATION
# ===============================
lemmatizer = WordNetLemmatizer()

lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

# ===============================
# 6. OUTPUT
# ===============================
print("Original Tokens:")
print(tokens)

print("\nLemmatized Tokens:")
print(lemmatized_words)