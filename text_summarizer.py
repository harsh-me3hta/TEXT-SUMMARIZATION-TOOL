import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import heapq

class TextSummarizer:
    def __init__(self):
        """
        Initialize the TextSummarizer with required NLTK resources
        """
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def preprocess_text(self, text):
        """
        Clean and preprocess the input text
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence endings
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text.strip()
    
    def extract_sentences(self, text):
        """
        Extract sentences from text using NLTK
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        sentences = sent_tokenize(text)
        return [sent.strip() for sent in sentences if len(sent.strip()) > 10]
    
    def extract_words(self, text):
        """
        Extract and preprocess words from text
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of processed words
        """
        words = word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        filtered_words = [
            self.stemmer.stem(word) for word in words 
            if word.isalpha() and word not in self.stop_words and len(word) > 2
        ]
        return filtered_words
    
    def calculate_sentence_scores_frequency(self, sentences):
        """
        Calculate sentence scores based on word frequency
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            dict: Dictionary with sentence indices as keys and scores as values
        """
        # Get word frequencies from all sentences
        all_words = []
        for sentence in sentences:
            words = self.extract_words(sentence)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Normalize frequencies
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = self.extract_words(sentence)
            score = 0
            for word in words:
                if word in word_freq:
                    score += word_freq[word]
            
            # Normalize by sentence length
            sentence_scores[i] = score / len(words) if words else 0
            
            # Positional bonus
            if i == 0:  # First sentence
                sentence_scores[i] *= 1.3
            if i == len(sentences) - 1:  # Last sentence
                sentence_scores[i] *= 1.1
                
            # Number presence bonus (often indicates facts/data)
            if re.search(r'\d+', sentence):
                sentence_scores[i] *= 1.2
        
        return sentence_scores
    
    def calculate_sentence_scores_tfidf(self, sentences):
        """
        Calculate sentence scores using TF-IDF vectorization
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            dict: Dictionary with sentence indices as keys and scores as values
        """
        if len(sentences) < 2:
            return {0: 1.0} if sentences else {}
        
        # Preprocess sentences for TF-IDF
        processed_sentences = []
        for sentence in sentences:
            words = self.extract_words(sentence)
            processed_sentences.append(' '.join(words))
        
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        
        # Calculate sentence scores as sum of TF-IDF values
        sentence_scores = {}
        for i in range(len(sentences)):
            sentence_scores[i] = np.sum(tfidf_matrix[i].toarray())
            
            # Apply positional and content bonuses
            if i == 0:
                sentence_scores[i] *= 1.3
            if i == len(sentences) - 1:
                sentence_scores[i] *= 1.1
            if re.search(r'\d+', sentences[i]):
                sentence_scores[i] *= 1.2
        
        return sentence_scores
    
    def calculate_sentence_similarity(self, sentences):
        """
        Calculate sentence similarity matrix using cosine similarity
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        if len(sentences) < 2:
            return np.array([[1.0]]) if sentences else np.array([])
        
        processed_sentences = []
        for sentence in sentences:
            words = self.extract_words(sentence)
            processed_sentences.append(' '.join(words))
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        return similarity_matrix
    
    def summarize_frequency_based(self, text, num_sentences=3):
        """
        Generate summary using frequency-based approach
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in summary
            
        Returns:
            dict: Summary result with text and metadata
        """
        if not text or len(text.strip()) < 50:
            raise ValueError("Text is too short to summarize. Please provide at least 50 characters.")
        
        text = self.preprocess_text(text)
        sentences = self.extract_sentences(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': ' '.join(sentences),
                'method': 'frequency_based',
                'original_sentences': len(sentences),
                'summary_sentences': len(sentences),
                'compression_ratio': 0
            }
        
        sentence_scores = self.calculate_sentence_scores_frequency(sentences)
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores.items(), key=lambda x: x[1])
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x[0])
        
        summary_sentences = [sentences[i] for i, score in top_sentences]
        summary = ' '.join(summary_sentences)
        
        return {
            'summary': summary,
            'method': 'frequency_based',
            'original_sentences': len(sentences),
            'summary_sentences': num_sentences,
            'compression_ratio': round((1 - num_sentences / len(sentences)) * 100, 1),
            'sentence_scores': sentence_scores
        }
    
    def summarize_tfidf_based(self, text, num_sentences=3):
        """
        Generate summary using TF-IDF approach
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in summary
            
        Returns:
            dict: Summary result with text and metadata
        """
        if not text or len(text.strip()) < 50:
            raise ValueError("Text is too short to summarize. Please provide at least 50 characters.")
        
        text = self.preprocess_text(text)
        sentences = self.extract_sentences(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': ' '.join(sentences),
                'method': 'tfidf_based',
                'original_sentences': len(sentences),
                'summary_sentences': len(sentences),
                'compression_ratio': 0
            }
        
        sentence_scores = self.calculate_sentence_scores_tfidf(sentences)
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, sentence_scores.items(), key=lambda x: x[1])
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x[0])
        
        summary_sentences = [sentences[i] for i, score in top_sentences]
        summary = ' '.join(summary_sentences)
        
        return {
            'summary': summary,
            'method': 'tfidf_based',
            'original_sentences': len(sentences),
            'summary_sentences': num_sentences,
            'compression_ratio': round((1 - num_sentences / len(sentences)) * 100, 1),
            'sentence_scores': sentence_scores
        }
    
    def summarize_hybrid(self, text, num_sentences=3):
        """
        Generate summary using hybrid approach (combining frequency and TF-IDF)
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in summary
            
        Returns:
            dict: Summary result with text and metadata
        """
        if not text or len(text.strip()) < 50:
            raise ValueError("Text is too short to summarize. Please provide at least 50 characters.")
        
        text = self.preprocess_text(text)
        sentences = self.extract_sentences(text)
        
        if len(sentences) <= num_sentences:
            return {
                'summary': ' '.join(sentences),
                'method': 'hybrid',
                'original_sentences': len(sentences),
                'summary_sentences': len(sentences),
                'compression_ratio': 0
            }
        
        # Get scores from both methods
        freq_scores = self.calculate_sentence_scores_frequency(sentences)
        tfidf_scores = self.calculate_sentence_scores_tfidf(sentences)
        
        # Normalize scores
        max_freq = max(freq_scores.values()) if freq_scores else 1
        max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1
        
        # Combine scores with weights
        combined_scores = {}
        for i in range(len(sentences)):
            freq_normalized = freq_scores.get(i, 0) / max_freq
            tfidf_normalized = tfidf_scores.get(i, 0) / max_tfidf
            combined_scores[i] = 0.6 * freq_normalized + 0.4 * tfidf_normalized
        
        # Get top sentences
        top_sentences = heapq.nlargest(num_sentences, combined_scores.items(), key=lambda x: x[1])
        
        # Sort by original order
        top_sentences.sort(key=lambda x: x[0])
        
        summary_sentences = [sentences[i] for i, score in top_sentences]
        summary = ' '.join(summary_sentences)
        
        return {
            'summary': summary,
            'method': 'hybrid',
            'original_sentences': len(sentences),
            'summary_sentences': num_sentences,
            'compression_ratio': round((1 - num_sentences / len(sentences)) * 100, 1),
            'sentence_scores': combined_scores
        }
    
    def summarize(self, text, num_sentences=3, method='hybrid'):
        """
        Main summarization method
        
        Args:
            text (str): Input text to summarize
            num_sentences (int): Number of sentences in summary
            method (str): Summarization method ('frequency', 'tfidf', 'hybrid')
            
        Returns:
            dict: Summary result with text and metadata
        """
        if method == 'frequency':
            return self.summarize_frequency_based(text, num_sentences)
        elif method == 'tfidf':
            return self.summarize_tfidf_based(text, num_sentences)
        else:  # hybrid
            return self.summarize_hybrid(text, num_sentences)

# Example usage and testing
if __name__ == "__main__":
    # Sample text for testing
    sample_text = """
    Artificial Intelligence (AI) has become one of the most transformative technologies of the 21st century. 
    Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy. 
    Natural Language Processing enables computers to understand and generate human language, while computer vision allows machines to interpret visual information. 
    These capabilities are revolutionizing industries from healthcare and finance to transportation and entertainment. 
    AI-powered systems can diagnose diseases, optimize financial portfolios, enable autonomous vehicles, and create personalized content recommendations. 
    However, the rapid advancement of AI also raises important ethical questions about privacy, job displacement, and the need for responsible development. 
    As we continue to integrate AI into our daily lives, it becomes crucial to ensure these technologies are developed and deployed in ways that benefit humanity while minimizing potential risks. 
    The future of AI promises even more exciting possibilities, including general artificial intelligence that could match or exceed human cognitive abilities across all domains. 
    Companies like OpenAI, Google, and Microsoft are investing billions of dollars in AI research and development. 
    The global AI market is expected to reach $1.8 trillion by 2030, according to recent industry reports.
    """
    
    # Initialize summarizer
    summarizer = TextSummarizer()
    
    # Test different methods
    print("=== FREQUENCY-BASED SUMMARY ===")
    result1 = summarizer.summarize(sample_text, num_sentences=3, method='frequency')
    print(f"Summary: {result1['summary']}")
    print(f"Compression: {result1['compression_ratio']}%")
    print()
    
    print("=== TF-IDF BASED SUMMARY ===")
    result2 = summarizer.summarize(sample_text, num_sentences=3, method='tfidf')
    print(f"Summary: {result2['summary']}")
    print(f"Compression: {result2['compression_ratio']}%")
    print()
    
    print("=== HYBRID SUMMARY ===")
    result3 = summarizer.summarize(sample_text, num_sentences=3, method='hybrid')
    print(f"Summary: {result3['summary']}")
    print(f"Compression: {result3['compression_ratio']}%")
    print()
