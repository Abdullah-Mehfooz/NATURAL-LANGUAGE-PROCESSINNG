# text_processing.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data (run once)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextProcessor:
    def __init__(self, remove_stopwords=True, lemmatize=True, stem=False):
        """
        Initialize Text Processor
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            stem: Whether to stem words (if True, overrides lemmatize)
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stem = stem
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stemmer = PorterStemmer() if stem else None
        
        # Add custom stopwords if needed
        self.custom_stopwords = {'rt', 'via', 'amp', '...', '…'}
        self.stop_words.update(self.custom_stopwords)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords_func(self, tokens):
        """Remove stopwords from tokens"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens):
        """Stem tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess(self, text, return_tokens=False):
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            return_tokens: If True, return tokens; if False, return string
        
        Returns:
            Processed text or tokens
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_func(tokens)
        
        # Apply stemming or lemmatization
        if self.stem and self.stemmer:
            tokens = self.stem_tokens(tokens)
        elif self.lemmatize and self.lemmatizer:
            tokens = self.lemmatize_tokens(tokens)
        
        # Return based on preference
        if return_tokens:
            return tokens
        else:
            return ' '.join(tokens)
    
    def preprocess_batch(self, texts, return_tokens=False):
        """Preprocess a list of texts"""
        return [self.preprocess(text, return_tokens) for text in texts]
    
    def analyze_text(self, text):
        """Analyze text and show processing steps"""
        print("Original Text:", text)
        print("Cleaned Text:", self.clean_text(text))
        
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        print("Tokens:", tokens)
        
        if self.remove_stopwords:
            filtered = self.remove_stopwords_func(tokens)
            print("After Stopword Removal:", filtered)
            tokens = filtered
        
        if self.stem and self.stemmer:
            stemmed = self.stem_tokens(tokens)
            print("After Stemming:", stemmed)
        elif self.lemmatize and self.lemmatizer:
            lemmatized = self.lemmatize_tokens(tokens)
            print("After Lemmatization:", lemmatized)


# Example usage
if __name__ == "__main__":
    # Sample texts
    texts = [
        "I LOVE this movie! It's amazing!!! Check it out: https://example.com",
        "RT @user: This is TERRIBLE! #worstmovieever",
        "The quick brown foxes are jumping over 10 lazy dogs.",
        "I'm running, he runs, they ran - all different forms!"
    ]
    
    # Initialize processor
    processor = TextProcessor(
        remove_stopwords=True,
        lemmatize=True,
        stem=False
    )
    
    print("=== Text Processing Example ===\n")
    
    # Analyze first text
    print("1. Text Analysis:")
    processor.analyze_text(texts[0])
    print()
    
    # Preprocess single text
    print("2. Single Text Processing:")
    processed = processor.preprocess(texts[0])
    print(f"Original: {texts[0]}")
    print(f"Processed: {processed}")
    print()
    
    # Batch processing
    print("3. Batch Processing:")
    processed_texts = processor.preprocess_batch(texts)
    for orig, proc in zip(texts, processed_texts):
        print(f"Original: {orig[:50]}...")
        print(f"Processed: {proc}")
        print()
    
    # Different configurations
    print("4. Different Processing Configurations:")
    configs = [
        ("No processing", TextProcessor(remove_stopwords=False, lemmatize=False, stem=False)),
        ("Stopwords only", TextProcessor(remove_stopwords=True, lemmatize=False, stem=False)),
        ("Lemmatization", TextProcessor(remove_stopwords=True, lemmatize=True, stem=False)),
        ("Stemming", TextProcessor(remove_stopwords=True, lemmatize=False, stem=True)),
    ]
    
    for config_name, proc_config in configs:
        result = proc_config.preprocess("The cats are running and jumping happily")
        print(f"{config_name}: {result}")