import pickle
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple, Set, Optional
import os
from .base_detector import BaseCandidateDetector, TransliterationCandidate
from ..utils.tokenizer import ThaiTokenizer

class NgramDetector(BaseCandidateDetector):
    """
    Statistical n-gram based detector for transliteration candidates
    """
    def __init__(self, 
                 model_path: Optional[str] = None,
                 tokenizer: Optional[ThaiTokenizer] = None,
                 name: str = "ngram_detector"):
        """
        Initialize the n-gram detector
        
        Args:
            model_path: Path to pre-trained n-gram model (if None, needs training)
            tokenizer: ThaiTokenizer instance (creates one if None)
            name: Name of the detector
        """
        super().__init__(name=name)
        self.tokenizer = tokenizer or ThaiTokenizer()
        
        # N-gram settings
        self.n_values = [2, 3, 4]  # Use bigrams, trigrams, and 4-grams
        
        # Model components
        self.thai_ngram_counts = {}
        self.trans_ngram_counts = {}
        self.total_thai_ngrams = {}
        self.total_trans_ngrams = {}
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            self.is_trained = True
        else:
            self.is_trained = False
            self.logger.warning("No model loaded. Train the model before using.")
    
    def train(self, 
              thai_texts: List[str], 
              transliterated_texts: List[str],
              save_path: Optional[str] = None):
        """
        Train the n-gram model
        
        Args:
            thai_texts: List of pure Thai texts
            transliterated_texts: List of texts with transliterations
            save_path: Path to save the trained model (optional)
        """
        self.logger.info("Training n-gram model...")
        
        # Extract n-grams from Thai texts
        for n in self.n_values:
            self.thai_ngram_counts[n] = Counter()
            
            for text in thai_texts:
                tokens = self.tokenizer.tokenize(text)
                for token in tokens:
                    ngrams = self._extract_character_ngrams(token, n)
                    self.thai_ngram_counts[n].update(ngrams)
                    
            self.total_thai_ngrams[n] = sum(self.thai_ngram_counts[n].values())
        
        # Extract n-grams from transliterated texts
        for n in self.n_values:
            self.trans_ngram_counts[n] = Counter()
            
            for text in transliterated_texts:
                tokens = self.tokenizer.tokenize(text)
                for token in tokens:
                    ngrams = self._extract_character_ngrams(token, n)
                    self.trans_ngram_counts[n].update(ngrams)
                    
            self.total_trans_ngrams[n] = sum(self.trans_ngram_counts[n].values())
        
        self.is_trained = True
        self.logger.info("N-gram model training completed")
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
    
    def save_model(self, path: str):
        """
        Save the trained model to disk
        
        Args:
            path: Path to save the model
        """
        model_data = {
            'thai_ngram_counts': self.thai_ngram_counts,
            'trans_ngram_counts': self.trans_ngram_counts,
            'total_thai_ngrams': self.total_thai_ngrams,
            'total_trans_ngrams': self.total_trans_ngrams,
            'n_values': self.n_values
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load a trained model from disk
        
        Args:
            path: Path to the model file
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
            
        self.thai_ngram_counts = model_data['thai_ngram_counts']
        self.trans_ngram_counts = model_data['trans_ngram_counts']
        self.total_thai_ngrams = model_data['total_thai_ngrams']
        self.total_trans_ngrams = model_data['total_trans_ngrams']
        self.n_values = model_data['n_values']
        
        self.logger.info(f"Model loaded from {path}")
    
    def detect_candidates(self, text: str) -> List[TransliterationCandidate]:
        """
        Detect transliteration candidates using n-gram statistics
        
        Args:
            text: Thai text to analyze
            
        Returns:
            List of transliteration candidates
        """
        if not self.is_trained:
            self.logger.error("Model not trained. Train or load a model first.")
            return []
        
        candidates = []
        
        # Tokenize with positions
        token_positions = self.tokenizer.tokenize_with_positions(text)
        
        for token, start_pos, end_pos in token_positions:
            # Skip very short tokens
            if len(token) < 2:
                continue
                
            confidence = self._calculate_ngram_confidence(token)
            
            if confidence > 0.5:  # Threshold can be adjusted
                metadata = {
                    "detection_method": "ngram_statistics",
                    "ngram_scores": {n: self._get_ngram_score(token, n) for n in self.n_values}
                }
                
                candidate = TransliterationCandidate(
                    token=token,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=confidence,
                    metadata=metadata
                )
                
                candidates.append(candidate)
                
        return candidates
    
    def _extract_character_ngrams(self, text: str, n: int) -> List[str]:
        """
        Extract character n-grams from text
        
        Args:
            text: Text to extract n-grams from
            n: Size of n-grams
            
        Returns:
            List of n-grams
        """
        # Pad the text for boundary n-grams
        padded = '_' + text + '_'
        return [padded[i:i+n] for i in range(len(padded)-n+1)]
    
    def _get_ngram_score(self, token: str, n: int) -> float:
        """
        Calculate n-gram score for a specific n value
        
        Args:
            token: Token to analyze
            n: Size of n-grams
            
        Returns:
            Score indicating likelihood of being a transliteration
        """
        ngrams = self._extract_character_ngrams(token, n)
        
        # Calculate log probability in Thai and transliteration distributions
        thai_prob = 0
        trans_prob = 0
        
        for ngram in ngrams:
            # Smoothed probability for Thai
            thai_count = self.thai_ngram_counts[n].get(ngram, 0) + 1  # Add-one smoothing
            thai_prob += np.log(thai_count / (self.total_thai_ngrams[n] + len(self.thai_ngram_counts[n])))
            
            # Smoothed probability for transliterations
            trans_count = self.trans_ngram_counts[n].get(ngram, 0) + 1  # Add-one smoothing
            trans_prob += np.log(trans_count / (self.total_trans_ngrams[n] + len(self.trans_ngram_counts[n])))
        
        # Return ratio of probabilities (higher means more likely to be transliteration)
        return 1 / (1 + np.exp(thai_prob - trans_prob))
    
    def _calculate_ngram_confidence(self, token: str) -> float:
        """
        Calculate overall confidence based on multiple n-gram sizes
        
        Args:
            token: Token to analyze
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Get scores for each n-gram size
        scores = [self._get_ngram_score(token, n) for n in self.n_values]
        
        # Weight larger n-grams more heavily
        weights = [0.2, 0.3, 0.5]  # Weights for 2-gram, 3-gram, 4-gram
        
        # Calculate weighted average
        weighted_score = sum(s * w for s, w in zip(scores, weights))
        
        return weighted_score