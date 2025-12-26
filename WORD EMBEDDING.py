from gensim.models import Word2Vec

# Tokenized sentences
sentences = [
    ["i", "love", "machine", "learning"],
    ["machine", "learning", "is", "fun"],
    ["i", "love", "ai"]
]

# Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=50,
    window=5,
    min_count=1,
    workers=4
)

# Vector of a word
vector = model.wv["machine"]
print("Vector size:", len(vector))
print(vector)

# Similar words
print("\nSimilar to 'machine':")
print(model.wv.most_similar("machine"))