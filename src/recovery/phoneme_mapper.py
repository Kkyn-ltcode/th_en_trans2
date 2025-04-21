import os
import json
import pickle
from typing import List, Dict, Tuple, Set, Optional, Any
import numpy as np
from collections import defaultdict

class PhonemeMapper:
    """
    Maps phonetic representations to potential English words
    """
    def __init__(self, 
                 english_phoneme_dict_path: Optional[str] = None,
                 similarity_threshold: float = 0.6):
        """
        Initialize the phoneme mapper
        
        Args:
            english_phoneme_dict_path: Path to English word-to-phoneme dictionary
            similarity_threshold: Threshold for phonetic similarity (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
        
        # Load English phoneme dictionary
        if english_phoneme_dict_path and os.path.exists(english_phoneme_dict_path):
            self._load_english_phoneme_dict(english_phoneme_dict_path)
        else:
            # Initialize empty dictionary
            self.english_phoneme_dict = {}
            self.phoneme_to_words = defaultdict(list)
            
        # Initialize phoneme similarity matrix
        self._initialize_phoneme_similarity()
    
    def _load_english_phoneme_dict(self, path: str):
        """
        Load English word-to-phoneme dictionary
        
        Args:
            path: Path to the dictionary file
        """
        # Determine file format based on extension
        _, ext = os.path.splitext(path)
        
        if ext.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                self.english_phoneme_dict = json.load(f)
        elif ext.lower() == '.pkl':
            with open(path, 'rb') as f:
                self.english_phoneme_dict = pickle.load(f)
        else:
            # Assume text format with word<tab>phonemes
            self.english_phoneme_dict = {}
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        word, phonemes = parts[0], parts[1]
                        self.english_phoneme_dict[word] = phonemes
        
        # Build reverse mapping (phoneme to words)
        self.phoneme_to_words = defaultdict(list)
        for word, phonemes in self.english_phoneme_dict.items():
            self.phoneme_to_words[phonemes].append(word)
    
    def _initialize_phoneme_similarity(self):
        """Initialize phoneme similarity matrix for fuzzy matching"""
        # This would be a comprehensive matrix of phoneme similarities
        # For example, 'p' and 'b' are similar, 'n' and 'm' are similar, etc.
        # For simplicity, we'll use a small example set
        
        self.phoneme_similarity = {
            # Consonant similarities
            ('p', 'b'): 0.8, ('t', 'd'): 0.8, ('k', 'g'): 0.8,
            ('f', 'v'): 0.8, ('s', 'z'): 0.8, ('ʃ', 'ʒ'): 0.8,
            ('m', 'n'): 0.7, ('n', 'ŋ'): 0.7,
            
            # Vowel similarities
            ('i', 'ɪ'): 0.9, ('u', 'ʊ'): 0.9, ('e', 'ɛ'): 0.9,
            ('o', 'ɔ'): 0.9, ('a', 'ɑ'): 0.9, ('a', 'æ'): 0.8,
            
            # Make similarity matrix symmetric
            ('b', 'p'): 0.8, ('d', 't'): 0.8, ('g', 'k'): 0.8,
            ('v', 'f'): 0.8, ('z', 's'): 0.8, ('ʒ', 'ʃ'): 0.8,
            ('n', 'm'): 0.7, ('ŋ', 'n'): 0.7,
            ('ɪ', 'i'): 0.9, ('ʊ', 'u'): 0.9, ('ɛ', 'e'): 0.9,
            ('ɔ', 'o'): 0.9, ('ɑ', 'a'): 0.9, ('æ', 'a'): 0.8,
        }
    
    def get_phoneme_similarity(self, phoneme1: str, phoneme2: str) -> float:
        """
        Get similarity score between two phonemes
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Exact match
        if phoneme1 == phoneme2:
            return 1.0
            
        # Known similarity
        if (phoneme1, phoneme2) in self.phoneme_similarity:
            return self.phoneme_similarity[(phoneme1, phoneme2)]
            
        # Default low similarity
        return 0.1
    
    def calculate_phoneme_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate similarity between two phoneme sequences
        
        Args:
            seq1: First phoneme sequence
            seq2: Second phoneme sequence
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Use dynamic programming to align sequences and calculate similarity
        # This is a simplified implementation of sequence alignment
        
        # Convert sequences to lists of phonemes
        phonemes1 = seq1.split('.')
        phonemes2 = seq2.split('.')
        
        # Initialize dynamic programming matrix
        m, n = len(phonemes1), len(phonemes2)
        dp = np.zeros((m + 1, n + 1))
        
        # Fill the matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                similarity = self.get_phoneme_similarity(phonemes1[i-1], phonemes2[j-1])
                dp[i, j] = max(
                    dp[i-1, j-1] + similarity,  # Match/substitution
                    dp[i-1, j] - 0.2,           # Deletion
                    dp[i, j-1] - 0.2            # Insertion
                )
        
        # Normalize by the length of the longer sequence
        max_length = max(m, n)
        if max_length == 0:
            return 0.0
            
        return dp[m, n] / max_length
    
    def find_english_candidates(self, 
                               thai_phonemes: str, 
                               max_candidates: int = 10) -> List[Tuple[str, float]]:
        """
        Find potential English word candidates for Thai phonemes
        
        Args:
            thai_phonemes: Phonetic representation of Thai word
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of (english_word, similarity_score) tuples
        """
        candidates = []
        
        # First, try exact phoneme matches
        if thai_phonemes in self.phoneme_to_words:
            exact_matches = self.phoneme_to_words[thai_phonemes]
            candidates.extend([(word, 1.0) for word in exact_matches])
        
        # If we don't have enough candidates, try fuzzy matching
        if len(candidates) < max_candidates:
            # Calculate similarity with all phonemes in the dictionary
            # This would be inefficient in practice - a real implementation would use
            # indexing or approximate nearest neighbor search
            
            # For demonstration, we'll limit to a small sample
            sample_size = min(1000, len(self.english_phoneme_dict))
            sample_words = list(self.english_phoneme_dict.keys())[:sample_size]
            
            for word in sample_words:
                english_phonemes = self.english_phoneme_dict[word]
                similarity = self.calculate_phoneme_sequence_similarity(thai_phonemes, english_phonemes)
                
                if similarity >= self.similarity_threshold:
                    candidates.append((word, similarity))
        
        # Sort by similarity score and take top candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:max_candidates]